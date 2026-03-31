from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge synthetic and real YOLO datasets into hybrid dataset.")
    parser.add_argument("--synth-root", type=Path, default=Path("boundary_ml/data/synth"))
    parser.add_argument("--real-root", type=Path, default=Path("boundary_ml/data/real"))
    parser.add_argument("--out-root", type=Path, default=Path("boundary_ml/data/hybrid"))
    return parser.parse_args()


def ensure_contract(root: Path, split: str) -> Tuple[Path, Path]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Expected pair not found: {image_dir} and {label_dir}. "
            "Create both folders and place matching image/txt files."
        )
    return image_dir, label_dir


def copy_split(prefix: str, image_dir: Path, label_dir: Path, out_root: Path, split: str) -> int:
    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    for pattern in exts:
        for image_path in image_dir.glob(pattern):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            out_name = f"{prefix}_{image_path.stem}"
            out_image = out_images / f"{out_name}{image_path.suffix.lower()}"
            out_label = out_labels / f"{out_name}.txt"

            shutil.copy2(image_path, out_image)
            shutil.copy2(label_path, out_label)
            count += 1

    return count


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
    args.out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ["train", "val"]:
        s_img, s_lbl = ensure_contract(args.synth_root, split)
        r_img, r_lbl = ensure_contract(args.real_root, split)

        synth_count = copy_split("synth", s_img, s_lbl, args.out_root, split)
        real_count = copy_split("real", r_img, r_lbl, args.out_root, split)
        total += synth_count + real_count

        print(f"{split}: synth={synth_count}, real={real_count}")

    write_yaml(args.out_root)
    print(f"Merged dataset ready: {args.out_root}")
    print(f"Total copied image/label pairs: {total}")


if __name__ == "__main__":
    main()
