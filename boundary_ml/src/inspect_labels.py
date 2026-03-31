from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize random YOLO labels.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("boundary_ml/data/synth"),
        help="Dataset root containing images/ and labels/ folders.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("boundary_ml/runs/inspect"),
        help="Where debug overlays are saved.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_label_line(line: str) -> List[float]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Bad label line: {line}")
    return [float(p) for p in parts]


def draw_label(image, label_file: Path) -> None:
    h, w = image.shape[:2]
    if not label_file.exists():
        return

    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, bw, bh = parse_label_line(line)
        if int(cls_id) != 0:
            continue

        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 235, 0), 2)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    images_dir = args.data_root / "images" / args.split
    labels_dir = args.data_root / "labels" / args.split

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected split folders not found in {args.data_root}")

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    picks = image_paths if len(image_paths) <= args.count else rng.sample(image_paths, args.count)
    for image_path in picks:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        draw_label(image, label_path)
        out_path = args.output_dir / f"{image_path.stem}_viz.jpg"
        cv2.imwrite(str(out_path), image)

    print(f"Saved {len(picks)} overlays to: {args.output_dir}")


if __name__ == "__main__":
    main()
