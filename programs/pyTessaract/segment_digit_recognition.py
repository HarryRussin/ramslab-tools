from __future__ import annotations

import cv2
import numpy as np


def _binarize_digit(image: np.ndarray) -> np.ndarray:
    """Ensure one digit crop is binary with white foreground on black background."""
    if image is None:
        raise ValueError("digit image is None")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary


def recognize_single_digit(image: np.ndarray) -> str:
    """Recognize one 7-segment digit from a binary crop."""
    binary = _binarize_digit(image)

    ys, xs = np.where(binary > 0)
    if ys.size == 0 or xs.size == 0:
        return "?"

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    fg_w = max(1, x1 - x0 + 1)
    fg_h = max(1, y1 - y0 + 1)
    aspect = fg_w / float(fg_h)

    # A very narrow, mostly vertical glyph is almost always a 1 on 7-segment clocks.
    if aspect < 0.34:
        roi = binary[y0 : y1 + 1, x0 : x1 + 1]
        split = max(1, int(roi.shape[1] * 0.45))
        left = roi[:, :split]
        right = roi[:, split:]
        left_density = float((left > 0).mean()) if left.size else 0.0
        right_density = float((right > 0).mean()) if right.size else 0.0
        if right_density >= left_density * 0.75:
            return "1"

    std_w, std_h = 60, 100
    resized = cv2.resize(binary, (std_w, std_h), interpolation=cv2.INTER_NEAREST)

    segments = {
        "a": resized[5:20, 15:45],
        "b": resized[18:45, 42:58],
        "c": resized[55:82, 42:58],
        "d": resized[80:95, 15:45],
        "e": resized[55:82, 2:18],
        "f": resized[18:45, 2:18],
        "g": resized[43:57, 15:45],
    }

    state = tuple(int((seg > 0).mean() > 0.24) for seg in segments.values())

    pattern_to_digit = {
        (1, 1, 1, 1, 1, 1, 0): "0",
        (0, 1, 1, 0, 0, 0, 0): "1",
        (1, 1, 0, 1, 1, 0, 1): "2",
        (1, 1, 1, 1, 0, 0, 1): "3",
        (0, 1, 1, 0, 0, 1, 1): "4",
        (1, 0, 1, 1, 0, 1, 1): "5",
        (1, 0, 1, 1, 1, 1, 1): "6",
        (1, 1, 1, 0, 0, 0, 0): "7",
        (1, 1, 1, 1, 1, 1, 1): "8",
        (1, 1, 1, 1, 0, 1, 1): "9",
    }

    if state in pattern_to_digit:
        return pattern_to_digit[state]

    best_digit = ""
    best_dist = 999
    for pattern, digit in pattern_to_digit.items():
        dist = sum(abs(a - b) for a, b in zip(state, pattern))
        if dist < best_dist:
            best_dist = dist
            best_digit = digit

    return best_digit if best_dist <= 2 else "?"


def recognize_digit_sequence(digit_images: list[np.ndarray]) -> list[str]:
    """Recognize ordered digit crops from left to right."""
    return [recognize_single_digit(img) for img in digit_images]


def format_clock_string(digits: list[str]) -> str:
    """Format recognized digits into a readable clock-like string."""
    clean = "".join(d for d in digits if d.isdigit())

    if len(clean) == 4:
        return f"{clean[:2]}:{clean[2:]}"
    if len(clean) == 3:
        return f"{clean[0]}:{clean[1:]}"
    if len(clean) == 2:
        return f"{clean[0]}:{clean[1]}"
    return clean


def recognize_and_format(digit_images: list[np.ndarray]) -> tuple[list[str], str]:
    """Recognize ordered digits and return a pretty assembled clock string."""
    digits = recognize_digit_sequence(digit_images)
    return digits, format_clock_string(digits)
