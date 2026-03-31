from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one-class YOLO detection data from cropped digit images.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("programs/data/data"),
        help="Folder with class subfolders 0..9 containing cropped digit images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("boundary_ml/data/synth"),
        help="Output folder for YOLO dataset.",
    )
    parser.add_argument("--train-count", type=int, default=2400)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--max-digits", type=int, default=5)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.2,
        help="Fraction of generated images with no digits.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def list_digit_images(source_root: Path) -> List[Path]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    images: List[Path] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name not in {str(i) for i in range(10)}:
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in exts:
                images.append(p)

    if not images:
        raise RuntimeError(f"No source images found under: {source_root}")
    return images


def make_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    mode = rng.choice(["solid", "gradient", "noise"])

    if mode == "solid":
        shade = rng.randint(80, 220)
        arr = np.full((height, width), shade, dtype=np.uint8)
    elif mode == "gradient":
        left = rng.randint(60, 200)
        right = rng.randint(60, 200)
        gradient = np.linspace(left, right, width, dtype=np.float32)
        arr = np.tile(gradient[None, :], (height, 1)).astype(np.uint8)
    else:
        arr = rng.randint(0, 255) * np.ones((height, width), dtype=np.uint8)
        noise = np.random.normal(loc=0.0, scale=rng.uniform(8.0, 24.0), size=(height, width))
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return cv2.GaussianBlur(arr, (5, 5), 0)


def extract_foreground(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]],
        axis=0,
    )
    bg_value = int(np.median(border))
    delta = np.abs(gray.astype(np.int16) - bg_value)
    mask = (delta > 18).astype(np.uint8) * 255

    if int(mask.sum()) < int(mask.size * 0.02):
        _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nonzero = int(mask_otsu.sum() / 255)
        inverted_nonzero = int((255 - mask_otsu).sum() / 255)
        mask = mask_otsu if nonzero < inverted_nonzero else (255 - mask_otsu)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray, np.ones_like(gray, dtype=np.uint8) * 255

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    crop = gray[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]

    kernel = np.ones((3, 3), np.uint8)
    crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return crop, crop_mask


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def find_position(
    canvas_w: int,
    canvas_h: int,
    box_w: int,
    box_h: int,
    placed: Sequence[Box],
    rng: random.Random,
) -> Box | None:
    if box_w >= canvas_w or box_h >= canvas_h:
        return None

    for _ in range(80):
        x = rng.randint(0, canvas_w - box_w)
        y = rng.randint(0, canvas_h - box_h)
        candidate = Box(x=x, y=y, w=box_w, h=box_h)
        if all(iou(candidate, other) < 0.1 for other in placed):
            return candidate
    return None


def blend_digit(
    background: np.ndarray,
    digit_crop: np.ndarray,
    digit_mask: np.ndarray,
    placement: Box,
    rng: random.Random,
) -> None:
    target_h = placement.h
    target_w = placement.w

    resized_digit = cv2.resize(digit_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(digit_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    if rng.random() < 0.5:
        resized_digit = 255 - resized_digit

    color = rng.randint(10, 245)
    painted = np.full_like(resized_digit, color, dtype=np.uint8)

    alpha = (resized_mask.astype(np.float32) / 255.0)[:, :, None]

    y1 = placement.y
    y2 = placement.y + target_h
    x1 = placement.x
    x2 = placement.x + target_w

    roi = background[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    digit_rgb = cv2.cvtColor(painted, cv2.COLOR_GRAY2BGR)

    blended = (alpha * digit_rgb + (1.0 - alpha) * roi_rgb).astype(np.uint8)
    background[y1:y2, x1:x2] = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)


def yolo_line(box: Box, width: int, height: int) -> str:
    x_center = (box.x + box.w / 2.0) / width
    y_center = (box.y + box.h / 2.0) / height
    w_norm = box.w / width
    h_norm = box.h / height
    return f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def build_one_image(
    source_images: Sequence[Path],
    width: int,
    height: int,
    max_digits: int,
    negative_ratio: float,
    rng: random.Random,
) -> Tuple[np.ndarray, List[Box]]:
    canvas = make_background(width=width, height=height, rng=rng)
    placements: List[Box] = []

    if rng.random() < negative_ratio:
        return canvas, placements

    count = rng.randint(1, max_digits)
    for _ in range(count):
        src = rng.choice(source_images)
        gray = np.array(Image.open(src).convert("L"))
        crop, mask = extract_foreground(gray)

        aspect = max(0.3, min(3.0, crop.shape[1] / max(1, crop.shape[0])))
        target_h = rng.randint(22, 120)
        target_w = int(target_h * aspect)
        target_w = max(14, min(target_w, 130))
        target_h = max(14, min(target_h, 130))

        pos = find_position(width, height, target_w, target_h, placements, rng)
        if pos is None:
            continue

        blend_digit(canvas, crop, mask, pos, rng)
        placements.append(pos)

    if rng.random() < 0.35:
        noise = np.random.normal(0, rng.uniform(2.0, 12.0), size=(height, width))
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        canvas = cv2.GaussianBlur(canvas, (3, 3), sigmaX=rng.uniform(0.1, 1.2))

    return canvas, placements


def ensure_dirs(root: Path) -> None:
    for rel in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def save_split(
    split: str,
    count: int,
    source_images: Sequence[Path],
    out_root: Path,
    width: int,
    height: int,
    max_digits: int,
    negative_ratio: float,
    rng: random.Random,
) -> None:
    for i in range(count):
        image, boxes = build_one_image(
            source_images=source_images,
            width=width,
            height=height,
            max_digits=max_digits,
            negative_ratio=negative_ratio,
            rng=rng,
        )

        image_name = f"{split}_{i:06d}.jpg"
        label_name = f"{split}_{i:06d}.txt"

        image_path = out_root / "images" / split / image_name
        label_path = out_root / "labels" / split / label_name

        cv2.imwrite(str(image_path), image)

        label_lines = [yolo_line(box, width=width, height=height) for box in boxes]
        label_path.write_text("\n".join(label_lines), encoding="utf-8")


def write_yaml(out_root: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {out_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names: ['digit']",
            "",
        ]
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    source_images = list_digit_images(args.source_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_dirs(args.output_root)

    save_split(
        split="train",
        count=args.train_count,
        source_images=source_images,
        out_root=args.output_root,
        width=args.width,
        height=args.height,
        max_digits=args.max_digits,
        negative_ratio=args.negative_ratio,
        rng=rng,
    )
    save_split(
        split="val",
        count=args.val_count,
        source_images=source_images,
        out_root=args.output_root,
        width=args.width,
        height=args.height,
        max_digits=args.max_digits,
        negative_ratio=args.negative_ratio,
        rng=rng,
    )

    write_yaml(args.output_root)

    total = args.train_count + args.val_count
    print(f"Generated {total} images at: {args.output_root}")
    print(f"YOLO config written to: {args.output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
