from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run digit-box inference and export plain coordinates.")
    parser.add_argument("--weights", type=Path, default=Path("boundary_ml/models/digit_detector_hybrid.pt"))
    parser.add_argument(
        "--source",
        type=str,
        default="boundary_ml/data/hybrid/images/val",
        help="Image file, folder, video, or glob source accepted by Ultralytics.",
    )
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=Path, default=Path("boundary_ml/runs"))
    parser.add_argument("--name", type=str, default="predict")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("boundary_ml/runs/predict/boxes.json"),
        help="JSON path for plain predicted boxes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    args.project.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        save=True,
        verbose=True,
    )

    payload: List[Dict[str, object]] = []
    for result in results:
        item: Dict[str, object] = {
            "source": str(result.path),
            "boxes": [],
        }

        if result.boxes is not None and result.boxes.xyxy is not None:
            xyxy = result.boxes.xyxy.cpu().numpy().tolist()
            confs = result.boxes.conf.cpu().numpy().tolist()
            for box, conf in zip(xyxy, confs):
                item["boxes"].append(
                    {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "conf": float(conf),
                    }
                )

        payload.append(item)

    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved plain boxes to: {args.out_json}")


if __name__ == "__main__":
    main()
