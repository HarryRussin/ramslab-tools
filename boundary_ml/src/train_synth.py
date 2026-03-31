from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-class digit detector on synthetic data.")
    parser.add_argument("--data", type=Path, default=Path("boundary_ml/data/synth/data.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("programs/scoreboard_site/yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", type=Path, default=Path("boundary_ml/runs"))
    parser.add_argument("--name", type=str, default="synth_train")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    args.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        optimizer="auto",
        verbose=True,
    )

    best_path = args.project / args.name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best_path}")

    models_dir = Path("boundary_ml/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "digit_detector_synth.pt"
    shutil.copy2(best_path, target)

    print(f"Best synth model copied to: {target}")


if __name__ == "__main__":
    main()
