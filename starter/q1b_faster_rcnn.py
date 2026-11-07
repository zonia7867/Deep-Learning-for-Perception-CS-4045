"""Question 1b starter - two-stage detector (Faster R-CNN, torchvision).

Dataset : the SAME 5-class split as Q1a, read straight out of data/voc_yolo/ so the
          comparison is apples to apples.
Model   : fasterrcnn_resnet50_fpn with COCO-pretrained weights. You replace the box
          predictor head for our class count and fine-tune. Nothing from scratch.

    python starter/q1b_faster_rcnn.py
    python starter/q1b_faster_rcnn.py --epochs 4 --trainable-layers 1   # cheaper run

Outputs in runs/q1/faster_rcnn/: results.json, predictions.png, best.pt

Label convention: torchvision detectors reserve id 0 for background, so our classes are
shifted to 1..N here, while YOLO uses 0..N-1. The evaluation below undoes the shift so
both detectors are scored on identical ground truth.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.common import count_parameters, get_device, load_config, save_results, set_seed
from utils.metrics import benchmark_latency, model_size_mb
from utils.viz import draw_boxes, save_image

ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = ROOT / "data" / "voc_yolo"


# --------------------------------------------------------------------------- data
class YoloFormatDetection(Dataset):
    """Reads the YOLO-format folders and yields torchvision detection targets.

    Returns (image_tensor, {"boxes": FloatTensor[N,4] xyxy, "labels": Int64Tensor[N]}).
    """

    def __init__(self, split: str, root: Path = YOLO_ROOT, train: bool = False) -> None:
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.files = sorted(self.img_dir.glob("*.jpg"))
        self.train = train
        if not self.files:
            raise SystemExit("no images in {}. Run setup/voc_to_yolo.py first.".format(self.img_dir))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        boxes, labels = [], []
        label_path = self.lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                cid, xc, yc, bw, bh = (float(v) for v in line.split())
                x1 = (xc - bw / 2) * width
                y1 = (yc - bh / 2) * height
                x2 = (xc + bw / 2) * width
                y2 = (yc + bh / 2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cid) + 1)        # +1: id 0 is background in torchvision

        image_t = TF.to_tensor(image)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if self.train and np.random.rand() < 0.5:  # paired horizontal flip
            image_t = torch.flip(image_t, dims=[2])
            if boxes_t.numel():
                x1 = boxes_t[:, 0].clone()
                boxes_t[:, 0] = width - boxes_t[:, 2]
                boxes_t[:, 2] = width - x1

        return image_t, {"boxes": boxes_t, "labels": labels_t,
                         "image_id": torch.tensor([idx])}


def collate(batch):
    return tuple(zip(*batch))


def class_names() -> list:
    with open(YOLO_ROOT / "data.yaml", "r", encoding="utf-8") as fh:
        names = yaml.safe_load(fh)["names"]
    return [names[i] for i in sorted(names)]


# -------------------------------------------------------------------------- model
def build_model(num_classes: int, trainable_layers: int = 3):
    """TODO(student) - swap the detection head.

    `fasterrcnn_resnet50_fpn` ships with a 91-class COCO box predictor. Our subset has
    len(class_names()) classes plus background, so the final predictor must be replaced
    before training; leave the backbone and the RPN pretrained.

    Do this:
        1. read the current input feature size:
               in_features = model.roi_heads.box_predictor.cls_score.in_features
        2. import FastRCNNPredictor from
               torchvision.models.detection.faster_rcnn
        3. assign a fresh head:
               model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    `num_classes` already includes background. In the report, explain what the two
    outputs of that predictor are (class scores and per-class box deltas) and why the
    RPN did NOT need replacing.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        trainable_backbone_layers=trainable_layers,   # 0 = frozen backbone, 5 = all
    ) 
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# -------------------------------------------------------------------------- loops
def train_one_epoch(model, loader, optimiser, device, epoch: int):
    model.train()
    running, batches = 0.0, 0
    for images, targets in tqdm(loader, leave=False, desc="train e{}".format(epoch)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items() if k != "image_id"}
                   for t in targets]
        losses = model(images, targets)               # in train mode it returns the loss dict
        loss = sum(losses.values())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimiser.step()
        running += float(loss)
        batches += 1
    return running / max(batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, score_threshold: float = 0.05):
    """mAP via torchmetrics, computed on the same boxes YOLO was scored on."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    for images, targets in tqdm(loader, leave=False, desc="eval"):
        images = [img.to(device) for img in images]
        outputs = model(images)
        preds, gts = [], []
        for out, tgt in zip(outputs, targets):
            keep = out["scores"] >= score_threshold
            preds.append({"boxes": out["boxes"][keep].cpu(),
                          "scores": out["scores"][keep].cpu(),
                          "labels": out["labels"][keep].cpu()})
            gts.append({"boxes": tgt["boxes"], "labels": tgt["labels"]})
        metric.update(preds, gts)
    return {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in metric.compute().items()}


@torch.no_grad()
def save_predictions(model, dataset, device, out_path, names, n: int = 4,
                     score_threshold: float = 0.5):
    model.eval()
    panels = []
    for i in range(min(n, len(dataset))):
        image_t, _ = dataset[i]
        out = model([image_t.to(device)])[0]
        keep = out["scores"] >= score_threshold
        image = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        panels.append(draw_boxes(image, out["boxes"][keep].cpu().numpy(),
                                 labels=[int(l) - 1 for l in out["labels"][keep].cpu()],
                                 scores=out["scores"][keep].cpu().numpy(),
                                 class_names=names))
    if not panels:
        return
    # VOC images are not all the same size, so pad every panel to the tallest before joining
    height = max(p.shape[0] for p in panels)
    padded = [np.pad(p, ((0, height - p.shape[0]), (0, 8), (0, 0)), constant_values=255)
              for p in panels]
    save_image(np.concatenate(padded, axis=1), out_path)


def main() -> None:
    cfg = load_config()
    q1 = cfg["q1_detection"]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=q1["faster_rcnn"]["epochs"])
    ap.add_argument("--batch", type=int, default=q1["faster_rcnn"]["batch_size"])
    ap.add_argument("--lr", type=float, default=q1["faster_rcnn"]["lr"])
    ap.add_argument("--trainable-layers", type=int, default=3, choices=[0, 1, 2, 3, 4, 5])
    args = ap.parse_args()

    set_seed(cfg.get("seed", 42))
    device = get_device()
    names = class_names()
    print("[device] {} | classes {}".format(device, names))

    train_set = YoloFormatDetection("train", train=True)
    val_set = YoloFormatDetection("val", train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=2, collate_fn=collate)
    print("[data] train {} | val {}".format(len(train_set), len(val_set)))

    model = build_model(num_classes=len(names) + 1,
                        trainable_layers=args.trainable_layers).to(device)
    total, trainable = count_parameters(model)
    print("[params] total {:,} | trainable {:,}".format(total, trainable))

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=3, gamma=0.1)

    out_dir = ROOT / "runs" / "q1" / "faster_rcnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_map50, best_stats = 0.0, None
    wall_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, device, epoch)
        stats = evaluate(model, val_loader, device)
        scheduler.step()
        print("epoch {:02d} | loss {:.3f} | mAP50 {:.4f} | mAP50-95 {:.4f}".format(
            epoch, train_loss, stats["map_50"], stats["map"]))
        if stats["map_50"] > best_map50:
            best_map50, best_stats = stats["map_50"], stats
            torch.save(model.state_dict(), out_dir / "best.pt")
    train_minutes = (time.perf_counter() - wall_start) / 60.0

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    save_predictions(model, val_set, device, out_dir / "predictions.png", names)
    speed = benchmark_latency(model, input_shape=(1, 3, q1["img_size"], q1["img_size"]),
                              device=device, warmup=5, iters=20)

    # torchmetrics reports map_per_class alongside a 'classes' tensor listing which label id
    # each entry belongs to - do not assume the order matches names[].
    per_class = {}
    if best_stats and isinstance(best_stats.get("map_per_class"), list):
        ids = best_stats.get("classes")
        ids = ids if isinstance(ids, list) else list(range(1, len(names) + 1))
        for class_id, value in zip(ids, best_stats["map_per_class"]):
            idx = int(class_id) - 1          # labels are 1..N, names are 0..N-1
            if 0 <= idx < len(names):
                per_class[names[idx]] = round(float(value), 4)

    save_results({
        "question": "Q1b",
        "detector": "faster_rcnn_resnet50_fpn",
        "epochs": args.epochs,
        "trainable_backbone_layers": args.trainable_layers,
        "imgsz": q1["img_size"],
        "mAP50": round(float(best_stats["map_50"]), 4),
        "mAP50_95": round(float(best_stats["map"]), 4),
        "mAP_small": round(float(best_stats.get("map_small", -1)), 4),
        "mAP_large": round(float(best_stats.get("map_large", -1)), 4),
        "per_class_AP50_95": per_class,
        "total_params": total,
        "trainable_params": trainable,
        "model_size_mb": model_size_mb(model),
        "inference": speed,
        "train_minutes": round(train_minutes, 1),
    }, ROOT / "runs" / "q1")

    print("\nNow run: python starter/make_tables.py   (builds the comparison tables)")


if __name__ == "__main__":
    main()
