from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_external_pytesseract():
	"""Load third-party pytesseract while avoiding local module shadowing."""
	current_dir = Path(__file__).resolve().parent
	original_sys_path = list(sys.path)

	try:
		sys.path = [p for p in sys.path if Path(p or ".").resolve() != current_dir]
		try:
			module = importlib.import_module("pytesseract")
		except Exception:
			return None
	finally:
		sys.path = original_sys_path

	if not hasattr(module, "image_to_string"):
		return None

	return module


tesseract_lib = _load_external_pytesseract()


# Optional override for Windows installs, e.g.:
# TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
if tesseract_lib is not None and os.environ.get("TESSERACT_CMD"):
	tesseract_lib.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


def _prep_for_tesseract(image):
	"""Light OCR prep for 7-segment digits: upscale + denoise + hard binary."""
	if image is None:
		raise ValueError("image is None")

	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Upscale to give Tesseract more stroke detail.
	upscaled = cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
	blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
	_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return binary


def _close_small_gaps(mask: np.ndarray, max_gap: int = 2) -> np.ndarray:
	"""Fill short false runs in a 1D boolean mask."""
	filled = mask.copy()
	idx = 0
	length = len(filled)
	while idx < length:
		if filled[idx]:
			idx += 1
			continue
		start = idx
		while idx < length and not filled[idx]:
			idx += 1
		end = idx
		if start > 0 and end < length and (end - start) <= int(max_gap):
			filled[start:end] = True
	return filled


def _extract_glyph_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
	"""Extract x-ordered glyph boxes from binary image via vertical projection."""
	h, w = binary.shape[:2]
	col_signal = (binary > 0).sum(axis=0) > max(1, int(h * 0.02))
	col_signal = _close_small_gaps(col_signal, max_gap=2)

	boxes: list[tuple[int, int, int, int]] = []
	x = 0
	while x < w:
		if not col_signal[x]:
			x += 1
			continue
		x1 = x
		while x < w and col_signal[x]:
			x += 1
		x2 = x
		if (x2 - x1) < 2:
			continue
		roi = binary[:, x1:x2]
		rows = np.where(roi.sum(axis=1) > 0)[0]
		if len(rows) == 0:
			continue
		y1 = int(rows.min())
		y2 = int(rows.max()) + 1
		boxes.append((x1, y1, x2 - x1, y2 - y1))

	return boxes


def _is_colon_roi(roi: np.ndarray) -> bool:
	"""Detect whether ROI looks like a two-dot colon."""
	num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
	components = []
	for idx in range(1, num_labels):
		area = int(stats[idx, cv2.CC_STAT_AREA])
		if area < 6:
			continue
		x = int(stats[idx, cv2.CC_STAT_LEFT])
		y = int(stats[idx, cv2.CC_STAT_TOP])
		w = int(stats[idx, cv2.CC_STAT_WIDTH])
		h = int(stats[idx, cv2.CC_STAT_HEIGHT])
		cx, cy = centroids[idx]
		components.append((x, y, w, h, area, float(cx), float(cy)))

	if len(components) != 2:
		return False

	c1, c2 = components
	dx = abs(c1[5] - c2[5])
	dy = abs(c1[6] - c2[6])
	# Colon dots should align vertically with significant y separation.
	return dx <= max(c1[2], c2[2], 4) and dy >= max(10, int(roi.shape[0] * 0.15))


def _decode_seven_segment_digit(roi: np.ndarray) -> str:
	"""Decode one digit ROI using 7-segment occupancy."""
	orig_h, orig_w = roi.shape[:2]
	std_w, std_h = 60, 100
	resized = cv2.resize(roi, (std_w, std_h), interpolation=cv2.INTER_NEAREST)
	_, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
	resized = cv2.morphologyEx(
		resized,
		cv2.MORPH_OPEN,
		cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
		iterations=1,
	)

	segments = {
		"a": resized[5:20, 15:45],
		"b": resized[18:45, 42:58],
		"c": resized[55:82, 42:58],
		"d": resized[80:95, 15:45],
		"e": resized[55:82, 2:18],
		"f": resized[18:45, 2:18],
		"g": resized[43:57, 15:45],
	}

	density = {name: float((seg > 0).mean()) for name, seg in segments.items()}
	state = tuple(int(density[name] > 0.24) for name in segments.keys())

	left_density = float((resized[:, : std_w // 2] > 0).mean())
	right_density = float((resized[:, std_w // 2 :] > 0).mean())
	horizontal_load = density["a"] + density["d"] + density["g"]
	left_rails = density["e"] + density["f"]
	right_rails = density["b"] + density["c"]

	if (
		right_density > (left_density * 1.35)
		and right_rails > 0.55
		and left_rails < 0.28
		and horizontal_load < 0.85
	):
		return "1"

	# Strong heuristic for digit "1": narrow glyph with dominant right rails.
	aspect = orig_w / float(max(1, orig_h))
	if (
		aspect <= 0.42
		and density["b"] > 0.22
		and density["c"] > 0.22
		and density["e"] < 0.12
		and density["f"] < 0.12
	):
		return "1"

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

	# Fallback: nearest known pattern by Hamming distance.
	best_digit = ""
	best_dist = 999
	for pattern, digit in pattern_to_digit.items():
		dist = sum(abs(a - b) for a, b in zip(state, pattern))
		if dist < best_dist:
			best_dist = dist
			best_digit = digit

	return best_digit if best_dist <= 2 else ""


def _read_clock_text_from_segments(binary: np.ndarray) -> tuple[str, str]:
	"""Pure OpenCV/Numpy 7-segment clock reader (no external OCR engine)."""
	boxes = _extract_glyph_boxes(binary)
	if not boxes:
		return "", ""

	glyphs = []
	for (x, y, w, h) in boxes:
		roi = binary[y : y + h, x : x + w]
		if _is_colon_roi(roi):
			glyphs.append(":")
			continue
		digit = _decode_seven_segment_digit(roi)
		if digit:
			glyphs.append(digit)

	raw = "".join(glyphs)
	normalized = normalize_clock_text(raw)
	return raw, normalized


def normalize_clock_text(text: str) -> str:
	"""Keep only digits and colon; normalize common OCR confusions."""
	if not text:
		return ""

	cleaned = text.strip().replace(" ", "")
	cleaned = cleaned.replace(";", ":").replace(".", ":")
	cleaned = re.sub(r"[^0-9:]", "", cleaned)

	# Collapse multiple colons to one and place between left/right digits when possible.
	if cleaned.count(":") > 1:
		parts = [p for p in cleaned.split(":") if p]
		if len(parts) >= 2:
			cleaned = f"{parts[0]}:{parts[1]}"
		else:
			cleaned = "".join(parts)

	if ":" not in cleaned and len(cleaned) >= 3:
		# Heuristic fallback for HHMM / HMM style outputs.
		if len(cleaned) == 4:
			cleaned = f"{cleaned[:2]}:{cleaned[2:]}"
		elif len(cleaned) == 3:
			cleaned = f"{cleaned[:1]}:{cleaned[1:]}"

	return cleaned


def read_clock_text_from_image(image, psm: int = 7) -> tuple[str, str]:
	"""
	Run pytesseract on a cropped time image.

	Returns:
		raw_text, normalized_text
	"""
	prepped = _prep_for_tesseract(image)

	if tesseract_lib is not None:
		try:
			config = f"--oem 3 --psm {int(psm)} -c tessedit_char_whitelist=0123456789:"
			raw = tesseract_lib.image_to_string(prepped, config=config)
			normalized = normalize_clock_text(raw)
			if normalized:
				return raw.strip(), normalized
		except Exception:
			pass

	# Fully self-contained fallback.
	return _read_clock_text_from_segments(prepped)


def read_clock_text_from_file(image_path: str | Path, psm: int = 7) -> tuple[str, str]:
	"""Load image from disk and run clock OCR."""
	path = Path(image_path)
	image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {path}")
	return read_clock_text_from_image(image, psm=psm)
