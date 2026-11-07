"""Question 2 starter - semantic segmentation with a pretrained encoder.

Dataset : Oxford-IIIT Pet trimaps. Every pixel is one of
          0 = pet, 1 = background, 2 = boundary (the original PNG stores 1/2/3; we remap).
Models  : U-Net with an ImageNet-pretrained encoder (segmentation_models_pytorch)
          DeepLabV3-ResNet50 with COCO-pretrained weights (torchvision)
Neither network is written from scratch - you assemble, fine-tune and evaluate them.

    python starter/q2_segmentation.py --model unet
    python starter/q2_segmentation.py --model unet --freeze-encoder     # ablation
    python starter/q2_segmentation.py --model unet --use-dice           # loss study
    python starter/q2_segmentation.py --model unet --encoder-weights none  # control
    python starter/q2_segmentation.py --model deeplabv3_resnet50

Outputs in runs/q2/<tag>/: results.json, curves.png, qualitative.png, best.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.common import count_parameters, get_device, load_config, save_results, set_seed
from utils.metrics import SegmentationMeter, benchmark_latency, model_size_mb
from utils.viz import plot_curves, show_segmentation_grid

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["pet", "background", "boundary"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- data
class PetSegmentation(Dataset):
    """Wraps torchvision's OxfordIIITPet so image and mask are resized consistently.

    The mask must be resized with NEAREST interpolation. Bilinear would invent class ids
    that do not exist (e.g. 1.5 between pet and boundary) and quietly corrupt your mIoU.
    """

    def __init__(self, split: str, img_size: int = 256, train: bool = False) -> None:
        self.base = OxfordIIITPet(root=str(ROOT / "data"), split=split,
                                  target_types="segmentation", download=False)
        self.img_size = img_size
        self.train = train
        self.normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]                      # PIL.Image, PIL.Image
        image = image.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.train and np.random.rand() < 0.5:         # paired horizontal flip
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_t = self.normalise(transforms.functional.to_tensor(image))
        mask_np = np.array(mask, dtype=np.int64) - 1      # 1,2,3 -> 0,1,2
        mask_np = np.clip(mask_np, 0, 2)
        return image_t, torch.from_numpy(mask_np)


def denormalise(image_t) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    arr = (image_t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


# -------------------------------------------------------------------------- model
def build_model(name: str, encoder: str, encoder_weights: str = "imagenet",
                num_classes: int = 3):
    if name == "unet":
        import segmentation_models_pytorch as smp

        weights = None if encoder_weights in ("none", None) else encoder_weights
        return smp.Unet(encoder_name=encoder, encoder_weights=weights,
                        in_channels=3, classes=num_classes)

    if name == "deeplabv3_resnet50":
        from torchvision import models

        weights = None if encoder_weights in ("none", None) else \
            models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = models.segmentation.deeplabv3_resnet50(weights=weights, aux_loss=True)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        return model

    raise ValueError("unknown model: {}".format(name))


def forward_logits(model, images):
    """torchvision segmentation models return a dict; smp models return a tensor."""
    out = model(images)
    return out["out"] if isinstance(out, dict) else out


def freeze_encoder(model, model_name: str) -> None:
    """TODO(student) 1 - the frozen-encoder ablation.

    Freeze only the pretrained encoder and leave the decoder / segmentation head
    trainable, then report mIoU and training time against the fully fine-tuned run.

    Where the encoder lives:
        unet (smp)          -> model.encoder
        deeplabv3_resnet50  -> model.backbone
    Two lines of code. State in the report how many parameters stayed trainable
    (utils.common.count_parameters prints it for you) and what that did to mIoU.
    """
    raise NotImplementedError("Implement freeze_encoder for the Q2.3 ablation.")


def dice_loss(logits, targets, num_classes: int = 3, eps: float = 1e-6):
    """Soft multi-class Dice loss. Provided - you only have to decide whether to use it."""
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * onehot).sum(dims)
    cardinality = probs.sum(dims) + onehot.sum(dims)
    return 1.0 - ((2 * intersection + eps) / (cardinality + eps)).mean()


def criterion_factory(use_dice: bool):
    """TODO(student) 2 - loss study.

    Run once with plain cross-entropy and once with cross-entropy + Dice
    (--use-dice). The boundary class is thin and heavily outnumbered, so the two
    losses do not treat it the same. Report per-class IoU for both and explain the
    difference you observe, especially for 'boundary'.
    """
    ce = nn.CrossEntropyLoss()

    def loss_fn(logits, targets):
        loss = ce(logits, targets)
        if use_dice:
            loss = loss + dice_loss(logits, targets)
        return loss

    return loss_fn


# -------------------------------------------------------------------------- loops
def train_one_epoch(model, loader, loss_fn, optimiser, device):
    model.train()
    running, seen = 0.0, 0
    for images, masks in tqdm(loader, leave=False, desc="train"):
        images, masks = images.to(device), masks.to(device)
        logits = forward_logits(model, images)
        loss = loss_fn(logits, masks)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        running += loss.item() * images.size(0)
        seen += images.size(0)
    return running / seen


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, collect_samples: int = 0):
    model.eval()
    meter = SegmentationMeter(num_classes=3)
    running, seen = 0.0, 0
    samples = []
    for images, masks in tqdm(loader, leave=False, desc="eval"):
        images, masks = images.to(device), masks.to(device)
        logits = forward_logits(model, images)
        running += loss_fn(logits, masks).item() * images.size(0)
        seen += images.size(0)
        preds = logits.argmax(1)
        meter.update(preds, masks)
        if len(samples) < collect_samples:
            samples.append((denormalise(images[0]),
                            masks[0].cpu().numpy(),
                            preds[0].cpu().numpy()))
    stats = meter.summary(CLASS_NAMES)
    stats["loss"] = running / seen
    return stats, samples


def main() -> None:
    cfg = load_config()
    q2 = cfg["q2_segmentation"]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=q2["model"], choices=["unet", "deeplabv3_resnet50"])
    ap.add_argument("--encoder", default=q2["encoder"], help="smp encoder name, e.g. resnet34")
    ap.add_argument("--encoder-weights", default="imagenet",
                    help="'imagenet' (or COCO for deeplab) | 'none' for the random-init control")
    ap.add_argument("--epochs", type=int, default=q2["epochs"])
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--use-dice", action="store_true")
    args = ap.parse_args()

    set_seed(cfg.get("seed", 42))
    device = get_device()
    img_size, batch_size = q2["img_size"], q2["batch_size"]
    print("[device] {}  img_size {}".format(device, img_size))

    train_set = PetSegmentation("trainval", img_size=img_size, train=True)
    test_set = PetSegmentation("test", img_size=img_size, train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    print("[data] train {} | test {}".format(len(train_set), len(test_set)))

    model = build_model(args.model, args.encoder, args.encoder_weights).to(device)
    if args.freeze_encoder:
        freeze_encoder(model, args.model)
    total, trainable = count_parameters(model)
    print("[params] total {:,} | trainable {:,}".format(total, trainable))

    loss_fn = criterion_factory(args.use_dice)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=q2["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    tag = "{}_{}{}{}".format(args.model, args.encoder_weights,
                             "_frozen" if args.freeze_encoder else "",
                             "_dice" if args.use_dice else "")
    out_dir = ROOT / "runs" / "q2" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou, best_stats = 0.0, None
    wall_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        stats, _ = evaluate(model, test_loader, loss_fn, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(stats["loss"])
        history["val_miou"].append(stats["mIoU"])
        print("epoch {:02d} | train {:.3f} | val {:.3f} | mIoU {:.4f} | per-class IoU {}".format(
            epoch, train_loss, stats["loss"], stats["mIoU"], stats["per_class_iou"]))
        if stats["mIoU"] > best_miou:
            best_miou, best_stats = stats["mIoU"], stats
            torch.save(model.state_dict(), out_dir / "best.pt")
    train_minutes = (time.perf_counter() - wall_start) / 60.0

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    _, samples = evaluate(model, test_loader, loss_fn, device, collect_samples=5)
    show_segmentation_grid(samples, out_dir / "qualitative.png")
    plot_curves(history, out_dir / "curves.png", title="Q2 {}".format(tag))

    speed = benchmark_latency(model, input_shape=(1, 3, img_size, img_size), device=device)

    save_results({
        "question": "Q2",
        "tag": tag,
        "model": args.model,
        "encoder": args.encoder,
        "encoder_weights": args.encoder_weights,
        "frozen_encoder": args.freeze_encoder,
        "loss": "ce+dice" if args.use_dice else "ce",
        "epochs": args.epochs,
        "img_size": img_size,
        "trainable_params": trainable,
        "total_params": total,
        "model_size_mb": model_size_mb(model),
        "best_mIoU": best_miou,
        "best_mDice": best_stats["mDice"],
        "pixel_accuracy": best_stats["pixel_accuracy"],
        "per_class_iou": best_stats["per_class_iou"],
        "train_minutes": round(train_minutes, 1),
        "inference": speed,
    }, ROOT / "runs" / "q2")

    print("\nQualitative grid saved. Pick two failure cases from it for your report and say")
    print("WHY they fail - thin boundary pixels, cluttered background, pet occluded, etc.")


if __name__ == "__main__":
    main()
