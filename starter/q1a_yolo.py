"""Question 1a starter - one-stage detector (YOLOv8, Ultralytics).

Dataset : the 5-class PASCAL VOC 2007 subset produced by setup/voc_to_yolo.py
Model   : yolov8n.pt - COCO-pretrained, fine-tuned on our subset. Nothing from scratch.

    python starter/q1a_yolo.py --zero-shot        # COCO weights, no training (baseline)
    python starter/q1a_yolo.py                    # fine-tune, then validate
    python starter/q1a_yolo.py --model yolov8s.pt # optional capacity comparison

Outputs: runs/q1/yolo/<tag>/ (Ultralytics writes its own plots there too:
confusion matrix, PR curve, label distribution - use them in your report)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.common import load_config, save_results, set_seed

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "voc_yolo" / "data.yaml"


def main() -> None:
    cfg = load_config()
    q1 = cfg["q1_detection"]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=q1["yolo"]["model"])
    ap.add_argument("--epochs", type=int, default=q1["yolo"]["epochs"])
    ap.add_argument("--batch", type=int, default=q1["yolo"]["batch_size"])
    ap.add_argument("--imgsz", type=int, default=q1["img_size"])
    ap.add_argument("--zero-shot", action="store_true",
                    help="evaluate the COCO-pretrained model without any fine-tuning")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("{} missing. Run: python setup/voc_to_yolo.py".format(DATA_YAML))

    from ultralytics import YOLO

    set_seed(cfg.get("seed", 42))
    tag = "{}_{}".format(Path(args.model).stem, "zeroshot" if args.zero_shot else
                         "ft{}".format(args.epochs))
    project = str(ROOT / "runs" / "q1" / "yolo")

    model = YOLO(args.model)
    train_minutes = 0.0

    if not args.zero_shot:
        # Ultralytics handles the training loop, augmentation, the anchor-free head and NMS.
        # Your job is to choose the settings, read the curves and explain the result.
        start = time.perf_counter()
        model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=cfg.get("seed", 42),
            project=project,
            name=tag,
            patience=10,
            pretrained=True,
            verbose=True,
        )
        train_minutes = (time.perf_counter() - start) / 60.0

    # Validation on our val split. For the zero-shot run the class ids come from COCO, so
    # the numbers are only meaningful for classes that exist in both label sets - say so.
    metrics = model.val(data=str(DATA_YAML), imgsz=args.imgsz, batch=args.batch,
                        project=project, name=tag + "_val", verbose=True)

    speed = getattr(metrics, "speed", {}) or {}
    inference_ms = float(speed.get("inference", 0.0))

    per_class = {}
    try:
        names = metrics.names if hasattr(metrics, "names") else {}
        for i, ap50_95 in enumerate(metrics.box.maps):
            per_class[str(names.get(i, i))] = round(float(ap50_95), 4)
    except Exception:
        pass

    record = {
        "question": "Q1a",
        "detector": "yolov8",
        "weights": args.model,
        "zero_shot": args.zero_shot,
        "epochs": 0 if args.zero_shot else args.epochs,
        "imgsz": args.imgsz,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class_AP50_95": per_class,
        "inference_ms_per_image": round(inference_ms, 2),
        "fps": round(1000.0 / inference_ms, 1) if inference_ms else None,
        "train_minutes": round(train_minutes, 1),
    }
    save_results(record, ROOT / "runs" / "q1")

    print("\nmAP@0.5 {:.4f} | mAP@0.5:0.95 {:.4f} | {:.1f} ms/image".format(
        record["mAP50"], record["mAP50_95"], inference_ms))
    print("Ultralytics also saved results.png, confusion_matrix.png and PR_curve.png under")
    print("   {}".format(Path(project) / tag))
    print("Next: python starter/q1b_faster_rcnn.py   (same data, two-stage detector)")


if __name__ == "__main__":
    main()
