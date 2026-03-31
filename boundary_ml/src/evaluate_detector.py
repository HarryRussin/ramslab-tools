from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate digit detector on a YOLO dataset.")
    parser.add_argument("--weights", type=Path, default=Path("boundary_ml/models/digit_detector_hybrid.pt"))
    parser.add_argument("--data", type=Path, default=Path("boundary_ml/data/hybrid/data.yaml"))
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=Path, default=Path("boundary_ml/runs"))
    parser.add_argument("--name", type=str, default="eval")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("boundary_ml/runs/eval/metrics.json"),
        help="JSON path where summarized metrics are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data}")

    args.project.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print(json.dumps(summary, indent=2))
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {args.out_json}")


if __name__ == "__main__":
    main()
