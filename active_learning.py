import os
import json
import random
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from transformers import SegformerConfig, SegformerForSemanticSegmentation
from pycocotools import mask as maskUtils



BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "dataset"
BEST_WEIGHTS = BASE_DIR / "best_segformer.pth"
OUT_DIR = BASE_DIR / "al_noisy_outputs"

BASELINE_BEST_VAL_DICE = 0.6599259149698312

SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_CHECKPOINT = "nvidia/segformer-b5-finetuned-ade-640-640"

IMAGE_SIZE = 256
TRAIN_BATCH_SIZE = 2     # безопасно для 4GB VRAM
VAL_BATCH_SIZE = 1

# Этап 0: обучить M_noisy на noisy-L0

# AMP + gradient accumulation
USE_AMP = (DEVICE == "cuda")
AMP_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
ACCUM_STEPS = 2 if TRAIN_BATCH_SIZE == 2 else 1  # accumulation при batch_size=2
NOISY_EPOCHS = 10#10
NOISY_LR = 5e-5
EARLY_STOP_PATIENCE = 3

# Итерации AL
L_FRAC = 0.30            # L0 ~ 30% от train
AL_ITERS = 3#4
K_PER_ITER = 20          # K на итерацию
FINE_TUNE_EPOCHS = 5#5
FINE_TUNE_LR = 2e-5

# Усиливаем вклад gold-поднабора (replay/oversample)
GOLD_REPEAT = 3

# "Плохая" разметка: уменьшение площади через повышение порога
NOISY_BLUR_KSIZE = 21     # должен быть нечетным
NOISY_THR = 0.85         # >0.5 => shrink
ERODE_KERNEL_SIZE = 5
ERODE_ITERATIONS = 2

POS_THR = 0.90
NEG_THR = 0.10

# Normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_annotation_file(split_dir: Path) -> Path:
    candidates = [
        split_dir / "annotations.coco.json",
        split_dir / "_annotations.coco.json",
        split_dir / "annotations.coco",
        split_dir / "_annotations.coco",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Не найден COCO-файл аннотаций в {split_dir}")


def load_coco(ann_path: Path) -> Dict:
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_coco_index(coco: Dict) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]]]:
    images = {im["id"]: im for im in coco.get("images", [])}
    anns_by_img: Dict[int, List[Dict]] = {k: [] for k in images.keys()}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id in anns_by_img:
            anns_by_img[img_id].append(ann)
    return images, anns_by_img


def resolve_image_path(images_dir: Path, file_name: str) -> Path:
    split_dir = images_dir.parent
    fn = (file_name or "").strip()
    if not fn:
        raise FileNotFoundError("Пустой file_name в COCO")

    fn = fn.replace("\\", "/")
    if fn.startswith("./"):
        fn = fn[2:]

    p = Path(fn)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)

    if fn.startswith("images/"):
        candidates.append(split_dir / Path(fn))
        candidates.append(images_dir / Path(fn[len("images/"):]))

    candidates.append(images_dir / Path(fn))
    candidates.append(images_dir / Path(fn).name)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Не найдено изображение '{file_name}' в {images_dir}")


def ann_to_binary_mask(ann: Dict, h: int, w: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if seg is None:
        return np.zeros((h, w), dtype=np.uint8)

    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return (m > 0).astype(np.uint8)

    if isinstance(seg, dict):
        m = maskUtils.decode(seg)
        return (m > 0).astype(np.uint8)

    return np.zeros((h, w), dtype=np.uint8)


def load_rgb_and_mask(split_dir: Path, image_info: Dict, anns: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Path]:
    images_dir = split_dir / "images"
    img_path = resolve_image_path(images_dir, image_info["file_name"])
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2.imread не смог прочитать: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        mask |= ann_to_binary_mask(ann, h, w)
    return rgb, mask, img_path


def all_img_ids(split_dir: Path) -> List[int]:
    coco = load_coco(find_annotation_file(split_dir))
    images_map, _ = build_coco_index(coco)
    return list(images_map.keys())

def make_noisy_mask_shrink(gt01: np.ndarray) -> np.ndarray:
    """
    gt01: uint8 0/1
    blur -> soft -> threshold NOISY_THR => shrink
    + erode (kernel=ERODE_KERNEL_SIZE, iterations=ERODE_ITERATIONS) for stronger shrink
    """
    gt = gt01.astype(np.float32)
    soft = cv2.GaussianBlur(gt, (NOISY_BLUR_KSIZE, NOISY_BLUR_KSIZE), sigmaX=0)
    noisy = (soft >= NOISY_THR).astype(np.uint8)

    if ERODE_ITERATIONS > 0:
        kernel = np.ones((ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE), dtype=np.uint8)
        noisy = cv2.erode(noisy, kernel, iterations=ERODE_ITERATIONS)

    return noisy

def preprocess_rgb(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    img = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img).float()


def resize_mask(mask01: np.ndarray, image_size: int) -> torch.Tensor:
    m = cv2.resize(mask01.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(m).float()

class SegFormerBinary(nn.Module):
    def __init__(self, init_from_hf: bool, hf_checkpoint: str):
        super().__init__()

        revision = os.getenv("HF_SEGFORMER_REVISION", "").strip() or None
        use_safetensors_env = os.getenv("HF_USE_SAFETENSORS", "").strip()

        if revision is None and hf_checkpoint == "nvidia/segformer-b5-finetuned-ade-640-640":
            revision = "refs/pr/3"
            if not use_safetensors_env:
                use_safetensors_env = "1"

        use_safetensors = None
        if use_safetensors_env:
            use_safetensors = use_safetensors_env.lower() not in ("0", "false", "no", "off")

        if init_from_hf:
            kwargs = dict(num_labels=1, ignore_mismatched_sizes=True)
            if revision is not None:
                kwargs["revision"] = revision
            if use_safetensors is True:
                kwargs["use_safetensors"] = True
            self.net = SegformerForSemanticSegmentation.from_pretrained(hf_checkpoint, **kwargs)
        else:
            cfg = SegformerConfig.from_pretrained(hf_checkpoint)
            cfg.num_labels = 1
            self.net = SegformerForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits



def _remap_best_segformer_state(state: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    model_keys = set(model_state.keys())
    out: Dict[str, torch.Tensor] = {}

    for k, v in state.items():
        if k in model_keys:
            out[k] = v
            continue

        candidates: List[str] = []

        k1 = re.sub(r"\.segformer\.encoder\.patch_embeddings\.(\d+)\.", r".segformer.stages.\1.patch_embeddings.", k)
        if k1 != k:
            candidates.append(k1)

        m = re.search(r"\.segformer\.encoder\.block\.(\d+)\.(\d+)\.", k)
        if m:
            i, j = m.group(1), m.group(2)
            base = re.sub(r"\.segformer\.encoder\.block\.(\d+)\.(\d+)\.", f".segformer.stages.{i}.{{KIND}}.{j}.", k)
            for kind in ("blocks", "block", "layers"):
                candidates.append(base.replace("{KIND}", kind))

        if ".segformer.encoder.layer_norm" in k:
            candidates.append(k.replace(".segformer.encoder.layer_norm", ".segformer.layer_norm"))

        for kk in candidates:
            if kk in model_keys:
                out[kk] = v
                break

    return out


def load_best_model_from_file(weights_path: Path) -> nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(f"Не найден файл весов: {weights_path}")

    model = SegFormerBinary(init_from_hf=False, hf_checkpoint=HF_CHECKPOINT)

    # torch.load: безопаснее грузить как веса (без произвольных объектов)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    model_state = model.state_dict()
    overlap = len(set(state.keys()) & set(model_state.keys()))

    remapped = _remap_best_segformer_state(state, model_state)
    if len(remapped) > overlap:
        state_to_load = remapped
    else:
        state_to_load = state

    res = model.load_state_dict(state_to_load, strict=False)

    loaded = len(state_to_load)
    total = len(model_state)
    load_ratio = loaded / max(total, 1)

    print(f"[BEST_WEIGHTS] overlap={overlap}, loaded_keys={loaded}/{total} (ratio={load_ratio:.3f}), "
          f"missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")

    if load_ratio < 0.5:
        print("[BEST_WEIGHTS][WARN] Похоже, что best_segformer.pth сохранён в другой версии transformers "
              "и большая часть весов не была загружена. Baseline метрики могут быть некорректны.")

    model.to(DEVICE)
    model.eval()
    return model


def init_m_noisy_from_hf() -> nn.Module:
    model = SegFormerBinary(init_from_hf=True, hf_checkpoint=HF_CHECKPOINT)
    model.to(DEVICE)
    model.train()
    return model

class SubsetDataset(Dataset):
    def __init__(self, split_dir: Path, image_size: int, img_ids: List[int], label_mode: str):
        assert label_mode in ("noisy", "gold", "unlabeled")
        self.split_dir = split_dir
        self.image_size = image_size
        self.label_mode = label_mode

        coco = load_coco(find_annotation_file(split_dir))
        self.images_map, self.anns_by_img = build_coco_index(coco)
        self.img_ids = img_ids

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        info = self.images_map[img_id]
        anns = self.anns_by_img.get(img_id, [])
        rgb, gt01, img_path = load_rgb_and_mask(self.split_dir, info, anns)

        x = preprocess_rgb(rgb, self.image_size)  # [3,S,S]
        name = img_path.name

        if self.label_mode == "unlabeled":
            y = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            valid = torch.zeros_like(y)
            return x, y, valid, name

        if self.label_mode == "gold":
            y01 = gt01
        else:
            y01 = make_noisy_mask_shrink(gt01)

        y = resize_mask(y01, self.image_size).unsqueeze(0)
        valid = torch.ones_like(y)
        return x, y, valid, name

def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss_map = loss_map * valid
    denom = valid.sum().clamp_min(1.0)
    return loss_map.sum() / denom


def masked_soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits) * valid
    targets = targets * valid
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    accum_steps: int,
    use_amp: bool,
) -> float:
    model.train()
    losses: List[float] = []

    optimizer.zero_grad(set_to_none=True)
    step_in_accum = 0

    for x, y, valid, _name in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        valid = valid.to(DEVICE)

        with autocast(device_type=AMP_DEVICE, enabled=use_amp):
            logits = model(x)
            loss = masked_bce_with_logits(logits, y, valid) + masked_soft_dice_loss(logits, y, valid)

        losses.append(float(loss.detach().cpu().item()))

        loss_to_backprop = loss / float(accum_steps)
        if use_amp:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        step_in_accum += 1
        if step_in_accum >= accum_steps:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0

    if step_in_accum > 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    losses, dices, ious = [], [], []
    for x, y, valid, _name in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        valid = valid.to(DEVICE)

        with autocast(device_type=AMP_DEVICE, enabled=USE_AMP):
            logits = model(x)
            loss = masked_bce_with_logits(logits, y, valid) + masked_soft_dice_loss(logits, y, valid)
        losses.append(float(loss.detach().cpu().item()))

        pred = (torch.sigmoid(logits) > 0.5).float()
        tp = (pred * y).sum().item()
        fp = (pred * (1-y)).sum().item()
        fn = ((1-pred) * y).sum().item()

        dices.append((2*tp) / (2*tp + fp + fn + 1e-6))
        ious.append(tp / (tp + fp + fn + 1e-6))

    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_dice": float(np.mean(dices)) if dices else 0.0,
        "val_iou": float(np.mean(ious)) if ious else 0.0,
    }


@torch.no_grad()
def eval_split_mean_std(model: nn.Module, split_dir: Path) -> Dict[str, float]:
    ids = all_img_ids(split_dir)
    ds = SubsetDataset(split_dir, IMAGE_SIZE, ids, label_mode="gold")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    dices, ious = [], []
    model.eval()
    for x, y, _valid, _name in dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float()

        tp = (pred * y).sum().item()
        fp = (pred * (1-y)).sum().item()
        fn = ((1-pred) * y).sum().item()

        dices.append((2*tp) / (2*tp + fp + fn + 1e-6))
        ious.append(tp / (tp + fp + fn + 1e-6))

    return {
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0,
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0,
        "n_images": int(len(ids)),
    }


# =========================================================
# AL uncertainty + selection
# =========================================================
@torch.no_grad()
def uncertainty_scores(model: nn.Module, loader: DataLoader) -> List[Tuple[str, float]]:
    model.eval()
    out = []
    for x, _y, _v, name in loader:
        x = x.to(DEVICE)
        with autocast(device_type=AMP_DEVICE, enabled=USE_AMP):
            logits = model(x)
            p = torch.sigmoid(logits)
        pos = (p >= POS_THR)
        neg = (p <= NEG_THR)
        confident = (pos | neg)
        cr = confident.float().mean(dim=(1,2,3)).item()
        out.append((name[0], 1.0 - cr))
    return out


def select_top_k(model: nn.Module, u_ds: SubsetDataset, k: int, iter_dir: Path) -> Tuple[List[str], List[Tuple[str, float]]]:
    iter_dir.mkdir(parents=True, exist_ok=True)
    sel_dir = iter_dir / "selected_images"
    sel_dir.mkdir(parents=True, exist_ok=True)

    u_dl = DataLoader(u_ds, batch_size=1, shuffle=False, num_workers=0)
    scores = uncertainty_scores(model, u_dl)
    scores.sort(key=lambda t: t[1], reverse=True)

    chosen = scores[:k] if k > 0 else scores
    chosen_names = [n for n, _ in chosen]

    with open(iter_dir / "selected.json", "w", encoding="utf-8") as f:
        json.dump([{"file": n, "uncertainty": s} for n, s in chosen], f, ensure_ascii=False, indent=2)

    # copy images
    images_dir = u_ds.split_dir / "images"
    for n in chosen_names:
        src = images_dir / n
        if not src.exists():
            matches = list(images_dir.rglob(n))
            if matches:
                src = matches[0]
        if src.exists():
            shutil.copy2(src, sel_dir / n)

    return chosen_names, scores


# =========================================================
# Training loop with early stopping (val_dice)
# =========================================================
def fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    tag: str,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler(device=AMP_DEVICE, enabled=USE_AMP)

    # baseline to prevent degradation relative to init
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_vm = validate(model, val_loader)
    best_val = best_vm["val_dice"]

    bad = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, ACCUM_STEPS, USE_AMP)
        vm = validate(model, val_loader)
        print(f"[{tag}] epoch={epoch} train_loss={tr_loss:.4f} val_loss={vm['val_loss']:.4f} val_dice={vm['val_dice']:.4f} val_iou={vm['val_iou']:.4f}")
        history.append({"epoch": float(epoch), "train_loss": float(tr_loss), "val_loss": float(vm['val_loss']), "val_dice": float(vm['val_dice']), "val_iou": float(vm['val_iou'])})

        if vm["val_dice"] > best_val + 1e-6:
            best_val = vm["val_dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            history: List[Dict[str, float]] = []
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state, strict=False)
    return model, history



# =========================================================
# Plotting helpers (saved to OUT_DIR/plots)
# =========================================================
def _ensure_plots_dir() -> Path:
    p = OUT_DIR / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_learning_curve(results: Dict, out_path: Path) -> None:
    iters = results["iterations"]
    xs = [d["iter"] for d in iters]
    val = [d["metrics"]["val"]["dice_mean"] for d in iters]
    rf  = [d["metrics"]["test_rf"]["dice_mean"] for d in iters]
    cv  = [d["metrics"]["test_cvat"]["dice_mean"] for d in iters]
    best_val = results["baseline_best_segformer"]["val"]["dice_mean"]

    plt.figure()
    plt.plot(xs, val, marker="o", label="val_dice")
    plt.plot(xs, rf,  marker="o", label="test_rf_dice")
    plt.plot(xs, cv,  marker="o", label="test_cvat_dice")
    plt.axhline(best_val, linestyle="--", label="best_segformer val_dice")
    plt.xlabel("AL iteration")
    plt.ylabel("Dice mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar_comparison(results: Dict, out_path: Path) -> None:
    iters = results["iterations"]
    labels = ["best_segformer", "M_noisy"] + [f"AL_{d['iter']}" for d in iters[1:]]
    vals = [results["baseline_best_segformer"]["val"]["dice_mean"], iters[0]["metrics"]["val"]["dice_mean"]] + \
           [d["metrics"]["val"]["dice_mean"] for d in iters[1:]]

    plt.figure()
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("val Dice mean")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_epoch_curves(results: Dict, out_dir: Path) -> None:
    hist = results.get("epoch_history", {})
    if not isinstance(hist, dict) or not hist:
        return

    for stage, rows in hist.items():
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        train_loss = [r["train_loss"] for r in rows]
        val_loss = [r["val_loss"] for r in rows]
        val_dice = [r["val_dice"] for r in rows]
        val_iou = [r["val_iou"] for r in rows]

        # loss chart
        plt.figure()
        plt.plot(epochs, train_loss, marker="o", label="train_loss")
        plt.plot(epochs, val_loss, marker="o", label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{stage}_loss_vs_epoch.png", dpi=200)
        plt.close()


        # metrics chart
        plt.figure()
        plt.plot(epochs, val_dice, marker="o", label="val_dice")
        plt.plot(epochs, val_iou, marker="o", label="val_iou")
        plt.xlabel("Эпоха")
        plt.ylabel("Метрика на валидации")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{stage}_metrics_vs_epoch.png", dpi=200)
        plt.close()


def plot_uncertainty_distributions(results: Dict, out_dir: Path) -> None:
    for d in results["iterations"]:
        it = d["iter"]
        scores = d.get("uncertainty_all", None)
        if not scores:
            continue
        vals = [float(s) for s in scores]
        plt.figure()
        plt.hist(vals, bins=20)
        plt.xlabel("uncertainty (1 - уверенность) ")
        plt.ylabel("Количество изображений")
        plt.title(f"U распределение uncertainty (итерация {it})")
        plt.tight_layout()
        plt.savefig(out_dir / f"uncertainty_hist_iter_{it:02d}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.boxplot(vals, vert=True)
        plt.ylabel("uncertainty (1 - уверенность) ")
        plt.title(f"U uncertainty boxplot (итерация {it})")
        plt.tight_layout()
        plt.savefig(out_dir / f"uncertainty_box_iter_{it:02d}.png", dpi=200)
        plt.close()


def plot_noisy_vs_gold_area(train_dir: Path, img_ids: List[int], out_dir: Path) -> None:
    coco = load_coco(find_annotation_file(train_dir))
    images_map, anns_by_img = build_coco_index(coco)

    gold_af, noisy_af = [], []
    for img_id in img_ids:
        info = images_map[img_id]
        anns = anns_by_img.get(img_id, [])
        rgb, gt01, _ = load_rgb_and_mask(train_dir, info, anns)
        h, w = gt01.shape[:2]
        gold = gt01.astype(np.float32)
        noisy = make_noisy_mask_shrink(gt01).astype(np.float32)
        gold_af.append(float(gold.mean()))
        noisy_af.append(float(noisy.mean()))

    # histogram
    plt.figure()
    plt.hist(gold_af, bins=20, alpha=0.6, label="gold")
    plt.hist(noisy_af, bins=20, alpha=0.6, label="noisy")
    plt.xlabel("Доля маски в изображении")
    plt.ylabel("Количество изображений")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mask_area_fraction_hist.png", dpi=200)
    plt.close()

    # boxplot
    plt.figure()
    plt.boxplot([gold_af, noisy_af], labels=["gold", "noisy"])
    plt.ylabel("Доля маски в изображении")
    plt.tight_layout()
    plt.savefig(out_dir / "mask_area_fraction_box.png", dpi=200)
    plt.close()


def _predict_mask_on_rgb(model: nn.Module, rgb: np.ndarray) -> np.ndarray:
    x = preprocess_rgb(rgb, IMAGE_SIZE).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with autocast(device_type=AMP_DEVICE, enabled=USE_AMP):
            logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float().squeeze(0).squeeze(0).detach().cpu().numpy()
    m = (pred > 0.5).astype(np.uint8) * 255
    return m


def save_qualitative_panels_iter1(model_before: nn.Module, model_after: nn.Module, train_dir: Path, selected_names: List[str], out_dir: Path, n_samples: int = 6) -> None:
    if not selected_names:
        return
    coco = load_coco(find_annotation_file(train_dir))
    images_map, anns_by_img = build_coco_index(coco)
    name_to_id = {Path(info["file_name"]).name: img_id for img_id, info in images_map.items()}

    chosen = selected_names[:n_samples]

    panels = []
    for name in chosen:
        img_id = name_to_id.get(name)
        if img_id is None:
            continue
        info = images_map[img_id]
        anns = anns_by_img.get(img_id, [])
        rgb, gt01, _ = load_rgb_and_mask(train_dir, info, anns)
        noisy01 = make_noisy_mask_shrink(gt01)

        # make visuals at 256x256
        img_vis = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        gold_vis = (cv2.resize(gt01*255, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST))
        noisy_vis = (cv2.resize(noisy01*255, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST))

        pred_b = _predict_mask_on_rgb(model_before, rgb)
        pred_a = _predict_mask_on_rgb(model_after, rgb)

        # convert masks to 3ch
        def m3(m): 
            return cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        row = np.concatenate([img_vis, m3(gold_vis), m3(noisy_vis), m3(pred_b), m3(pred_a)], axis=1)

        # add labels
        labels = ["image", "gold", "noisy", "pred_before", "pred_after"]
        x0 = 5
        for j, lab in enumerate(labels):
            cv2.putText(row, lab, (j*IMAGE_SIZE + x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        panels.append(row)

    if not panels:
        return

    collage = np.concatenate(panels, axis=0)
    out_path = out_dir / "qualitative_iter01.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))

def main() -> None:
    seed_everything(SEED)
    print(f"[CONFIG] DEVICE={DEVICE} USE_AMP={USE_AMP} TRAIN_BATCH_SIZE={TRAIN_BATCH_SIZE} ACCUM_STEPS={ACCUM_STEPS} EFFECTIVE_BATCH≈{TRAIN_BATCH_SIZE*ACCUM_STEPS} IMAGE_SIZE={IMAGE_SIZE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = DATASET_ROOT / "train"
    val_dir = DATASET_ROOT / "val"
    test_rf_dir = DATASET_ROOT / "test_rf"
    test_cvat_dir = DATASET_ROOT / "test_cvat"

    for p in (train_dir, val_dir, test_rf_dir, test_cvat_dir):
        if not p.exists():
            raise FileNotFoundError(f"Не найдена папка: {p}")

    baseline = {
        "val": {"dice_mean": float(BASELINE_BEST_VAL_DICE)},
        "test_rf": None,
        "test_cvat": None,
    }

    all_ids = sorted(all_img_ids(train_dir))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_ids))
    L0_size = int(round(L_FRAC * len(all_ids)))

    L_noisy = [all_ids[i] for i in perm[:L0_size]]
    U_pool = [all_ids[i] for i in perm[L0_size:]]
    L_gold: Set[int] = set()

    val_ids = sorted(all_img_ids(val_dir))
    val_ds = SubsetDataset(val_dir, IMAGE_SIZE, val_ids, label_mode="gold")
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)

    m = init_m_noisy_from_hf()
    l0_ds = SubsetDataset(train_dir, IMAGE_SIZE, L_noisy, label_mode="noisy")
    l0_loader = DataLoader(l0_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

    m, hist0 = fit_with_early_stopping(
        m,
        l0_loader,
        val_loader,
        lr=NOISY_LR,
        max_epochs=NOISY_EPOCHS,
        patience=EARLY_STOP_PATIENCE,
        tag="M_noisy_init",
    )

    def eval_pack(model: nn.Module) -> Dict[str, Dict[str, float]]:
        return {
            "val": eval_split_mean_std(model, val_dir),
            "test_rf": eval_split_mean_std(model, test_rf_dir),
            "test_cvat": eval_split_mean_std(model, test_cvat_dir),
        }

    results: Dict[str, object] = {
        "baseline_best_segformer": baseline,
        "config": {
            "seed": SEED,
            "L_FRAC": L_FRAC,
            "L0_size": len(L_noisy),
            "U0_size": len(U_pool),
            "NOISY_THR": NOISY_THR,
            "NOISY_BLUR_KSIZE": NOISY_BLUR_KSIZE,
            "POS_THR": POS_THR,
            "NEG_THR": NEG_THR,
            "AL_ITERS": AL_ITERS,
            "K_PER_ITER": K_PER_ITER,
            "GOLD_REPEAT": GOLD_REPEAT,
            "IMAGE_SIZE": IMAGE_SIZE,
            "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
            "NOISY_EPOCHS": NOISY_EPOCHS,
            "FINE_TUNE_EPOCHS": FINE_TUNE_EPOCHS,
            "NOISY_LR": NOISY_LR,
            "FINE_TUNE_LR": FINE_TUNE_LR,
        },
        "iterations": [],
    }

    results.setdefault("epoch_history", {})["M_noisy_init"] = hist0

    # Iteration 0 metrics
    results["iterations"].append({
        "iter": 0,
        "L_noisy": len(L_noisy),
        "L_gold": len(L_gold),
        "U_pool": len(U_pool),
        "selected": [],
        "uncertainty_all": [],
        "metrics": eval_pack(m),
    })

    for it in range(1, AL_ITERS + 1):
        if len(U_pool) == 0:
            break

        iter_dir = OUT_DIR / f"iter_{it:02d}"

        u_ds = SubsetDataset(train_dir, IMAGE_SIZE, U_pool, label_mode="unlabeled")
        chosen_names, all_scores = select_top_k(m, u_ds, min(K_PER_ITER, len(U_pool)), iter_dir)

        coco = load_coco(find_annotation_file(train_dir))
        images_map, _ = build_coco_index(coco)
        name_to_id = {Path(info["file_name"]).name: img_id for img_id, info in images_map.items()}

        chosen_ids = []
        for n in chosen_names:
            img_id = name_to_id.get(n)
            if img_id is not None and img_id in U_pool:
                chosen_ids.append(img_id)

        for img_id in chosen_ids:
            U_pool.remove(img_id)
            L_gold.add(img_id)

        noisy_ds = SubsetDataset(train_dir, IMAGE_SIZE, L_noisy, label_mode="noisy")
        gold_ds = SubsetDataset(train_dir, IMAGE_SIZE, sorted(list(L_gold)), label_mode="gold")

        parts = [noisy_ds] + [gold_ds] * max(1, GOLD_REPEAT)

        class Concat(Dataset):
            def __init__(self, ds_list):
                self.ds_list = ds_list
                self.cum = []
                s = 0
                for d in ds_list:
                    s += len(d)
                    self.cum.append(s)
            def __len__(self): return self.cum[-1] if self.cum else 0
            def __getitem__(self, idx):
                for i, c in enumerate(self.cum):
                    if idx < c:
                        prev = 0 if i == 0 else self.cum[i-1]
                        return self.ds_list[i][idx - prev]
                raise IndexError(idx)

        ft_ds = Concat(parts)
        ft_loader = DataLoader(ft_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

        # сохраняем модель до дообучения для qualitative panel на итерации 1
        model_before = None
        if it == 1:
            model_before = SegFormerBinary(init_from_hf=False, hf_checkpoint=HF_CHECKPOINT)
            model_before.load_state_dict({k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, strict=False)
            model_before.to(DEVICE)
            model_before.eval()

        m, hist_it = fit_with_early_stopping(
            m,
            ft_loader,
            val_loader,
            lr=FINE_TUNE_LR,
            max_epochs=FINE_TUNE_EPOCHS,
            patience=EARLY_STOP_PATIENCE,
            tag=f"AL_iter_{it}",
        )
        results.setdefault("epoch_history", {})[f"AL_iter_{it}"] = hist_it

        if it == 1 and model_before is not None:
            plots_dir = _ensure_plots_dir()
            save_qualitative_panels_iter1(model_before, m, train_dir, chosen_names, plots_dir, n_samples=6)

        results["iterations"].append({
            "iter": it,
            "L_noisy": len(L_noisy),
            "L_gold": len(L_gold),
            "U_pool": len(U_pool),
            "selected": chosen_names,
            "uncertainty_all": [float(s) for _n, s in all_scores],
            "metrics": eval_pack(m),
        })

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plots_dir = _ensure_plots_dir()
    plot_learning_curve(results, plots_dir / "learning_curve_dice.png")
    plot_bar_comparison(results, plots_dir / "bar_val_dice.png")
    plot_epoch_curves(results, plots_dir)
    plot_uncertainty_distributions(results, plots_dir)
    plot_noisy_vs_gold_area(train_dir, all_ids, plots_dir)
    print("[PLOTS] saved to:", plots_dir)

    print("Done. Results saved:", out_path)
    print("Baseline (best_segformer) val dice:", baseline["val"]["dice_mean"])
    print("M_noisy final val dice:", results["iterations"][-1]["metrics"]["val"]["dice_mean"])


if __name__ == "__main__":
    main()
