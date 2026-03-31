from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


Rect = Tuple[int, int, int, int]


def _to_binary_white_foreground(image: np.ndarray, threshold: int = 127) -> np.ndarray:
	"""Return a binary image with white foreground (255) and black background (0)."""
	if image is None:
		raise ValueError("image is None")

	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	_, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
	return binary


def _pad_rect(rect: Rect, pad_px: int, width: int, height: int) -> Rect:
	x, y, w, h = rect
	x1 = max(0, x - pad_px)
	y1 = max(0, y - pad_px)
	x2 = min(width, x + w + pad_px)
	y2 = min(height, y + h + pad_px)
	return x1, y1, x2 - x1, y2 - y1


def _rect_gap(a: Rect, b: Rect) -> int:
	"""Return edge-to-edge pixel gap between two rectangles (0 means touching/overlapping)."""
	ax, ay, aw, ah = a
	bx, by, bw, bh = b

	ax2 = ax + aw
	ay2 = ay + ah
	bx2 = bx + bw
	by2 = by + bh

	dx = max(0, max(bx - ax2, ax - bx2))
	dy = max(0, max(by - ay2, ay - by2))
	return int(max(dx, dy))


def _union_rects(rects: list[Rect]) -> Rect:
	"""Return the bounding rectangle enclosing all input rectangles."""
	x1 = min(r[0] for r in rects)
	y1 = min(r[1] for r in rects)
	x2 = max(r[0] + r[2] for r in rects)
	y2 = max(r[1] + r[3] for r in rects)
	return x1, y1, x2 - x1, y2 - y1


def _rect_center_x(rect: Rect) -> float:
	x, _, w, _ = rect
	return x + (w / 2.0)


def _extract_component_rects(binary_image: np.ndarray, min_area: int = 5) -> list[Rect]:
	"""Extract connected-component bounding boxes above a minimum area."""
	num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
	rects: list[Rect] = []
	for idx in range(1, num_labels):
		x = int(stats[idx, cv2.CC_STAT_LEFT])
		y = int(stats[idx, cv2.CC_STAT_TOP])
		w = int(stats[idx, cv2.CC_STAT_WIDTH])
		h = int(stats[idx, cv2.CC_STAT_HEIGHT])
		area = int(stats[idx, cv2.CC_STAT_AREA])
		if area >= int(min_area):
			rects.append((x, y, w, h))
	return rects


def _find_colon_rect(binary_image: np.ndarray) -> Rect | None:
	"""
	Find a likely colon made of two vertically aligned dot components.

	Returns colon bounding rectangle or None.
	"""
	h, w = binary_image.shape[:2]
	all_rects = _extract_component_rects(binary_image, min_area=3)

	# Colon dots are small compared to digits.
	max_dot_area = max(8, int((h * w) * 0.0025))
	candidates: list[tuple[Rect, int]] = []
	for rect in all_rects:
		x, y, rw, rh = rect
		area = rw * rh
		if area <= max_dot_area:
			candidates.append((rect, area))

	best_pair_rect: Rect | None = None
	best_pair_score = -1.0

	for i in range(len(candidates)):
		r1, a1 = candidates[i]
		x1, y1, w1, h1 = r1
		cx1 = x1 + (w1 / 2.0)
		cy1 = y1 + (h1 / 2.0)
		for j in range(i + 1, len(candidates)):
			r2, a2 = candidates[j]
			x2, y2, w2, h2 = r2
			cx2 = x2 + (w2 / 2.0)
			cy2 = y2 + (h2 / 2.0)

			dx = abs(cx1 - cx2)
			dy = abs(cy1 - cy2)
			if dx > max(6, int(w * 0.02)):
				continue
			if dy < max(8, int(h * 0.02)) or dy > max(80, int(h * 0.30)):
				continue

			area_similarity = min(a1, a2) / float(max(a1, a2))
			width_similarity = min(w1, w2) / float(max(w1, w2))
			score = (area_similarity * 0.6) + (width_similarity * 0.4)
			if score > best_pair_score:
				best_pair_score = score
				best_pair_rect = _union_rects([r1, r2])

	return best_pair_rect


def _grow_neighbor_cluster(seed_idx: int, rects: list[Rect], neighbor_distance_px: int) -> list[Rect]:
	"""Grow a rectangle cluster recursively from seed using neighbor gap distance."""
	visited = {seed_idx}
	queue = [seed_idx]
	cluster = [rects[seed_idx]]

	while queue:
		current = queue.pop(0)
		current_rect = rects[current]
		for idx, rect in enumerate(rects):
			if idx in visited:
				continue
			if _rect_gap(current_rect, rect) > int(neighbor_distance_px):
				continue
			visited.add(idx)
			queue.append(idx)
			cluster.append(rect)

	return cluster


def find_highest_signal_rect(
	binary_image: np.ndarray,
	min_area: int = 150,
	neighbor_distance_px: int = 0,
	high_density_ratio: float = 0.60,
) -> Rect:
	"""
	Find rectangle with the highest percentage of white pixels.

	The rectangle candidates come from connected-component bounding boxes.
	Score = white_pixels_in_box / box_area.
	"""
	if binary_image is None:
		raise ValueError("binary_image is None")

	height, width = binary_image.shape[:2]
	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	candidates: list[tuple[Rect, float]] = []

	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		x = int(x)
		y = int(y)
		w = int(w)
		h = int(h)
		area = w * h

		if area < int(min_area):
			continue

		roi = binary_image[y : y + h, x : x + w]
		white_count = int(cv2.countNonZero(roi))
		score = white_count / float(area)
		candidates.append(((x, y, w, h), score))

	if candidates:
		best_idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
		best_rect, best_score = candidates[best_idx]

		if int(neighbor_distance_px) <= 0:
			return best_rect

		min_neighbor_score = best_score * float(high_density_ratio)
		visited = {best_idx}
		queue = [best_idx]
		cluster_rects = [best_rect]

		# Recursively include neighboring high-density boxes.
		while queue:
			current_idx = queue.pop(0)
			current_rect, _ = candidates[current_idx]

			for idx, (rect, score) in enumerate(candidates):
				if idx in visited:
					continue
				if score < min_neighbor_score:
					continue
				if _rect_gap(current_rect, rect) > int(neighbor_distance_px):
					continue

				visited.add(idx)
				queue.append(idx)
				cluster_rects.append(rect)

		return _union_rects(cluster_rects)

	ys, xs = np.where(binary_image > 0)
	if len(xs) > 0 and len(ys) > 0:
		x1 = int(xs.min())
		x2 = int(xs.max())
		y1 = int(ys.min())
		y2 = int(ys.max())
		return x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)

	return 0, 0, width, height


def crop_highest_signal_region(
	image: np.ndarray,
	processed_signal_image: np.ndarray | None = None,
	threshold: int = 127,
	min_area: int = 150,
	neighbor_distance_px: int = 0,
	pad_px: int = 8,
) -> tuple[np.ndarray, Rect]:
	"""
	Crop to the rectangle with highest white-pixel percentage and apply padding.

	If processed_signal_image is provided, it is used as the signal map
	for rectangle selection. This is useful when a prior preprocessing step
	already produced a binary/near-binary OCR image.

	Returns:
		cropped_image, (x, y, w, h)
	"""
	if image is None:
		raise ValueError("image is None")

	if processed_signal_image is None:
		binary = _to_binary_white_foreground(image, threshold=threshold)
	else:
		binary = _to_binary_white_foreground(processed_signal_image, threshold=threshold)

	rect = find_highest_signal_rect(
		binary,
		min_area=min_area,
		neighbor_distance_px=neighbor_distance_px,
	)

	h, w = image.shape[:2]
	padded_rect = _pad_rect(rect, pad_px=pad_px, width=w, height=h)
	x, y, rw, rh = padded_rect

	cropped = image[y : y + rh, x : x + rw]
	return cropped, padded_rect


def extract_top_signal_regions(
	image: np.ndarray,
	processed_signal_image: np.ndarray | None = None,
	threshold: int = 127,
	min_area: int = 150,
	neighbor_distance_px: int = 0,
	pad_px: int = 8,
	max_regions: int = 2,
	min_white_pixels: int = 20,
) -> list[tuple[np.ndarray, Rect]]:
	"""
	Iteratively extract top signal regions by masking each selected box and rerunning.

	After each region is selected, the same padded rectangle is set to black in the
	working binary signal image so the next call finds another high-density area.
	"""
	if image is None:
		raise ValueError("image is None")
	if max_regions <= 0:
		return []

	if processed_signal_image is None:
		binary = _to_binary_white_foreground(image, threshold=threshold)
	else:
		binary = _to_binary_white_foreground(processed_signal_image, threshold=threshold)

	working_binary = binary.copy()
	h, w = image.shape[:2]
	results: list[tuple[np.ndarray, Rect]] = []

	for _ in range(int(max_regions)):
		if int(cv2.countNonZero(working_binary)) < int(min_white_pixels):
			break

		rect = find_highest_signal_rect(
			working_binary,
			min_area=min_area,
			neighbor_distance_px=neighbor_distance_px,
		)
		padded_rect = _pad_rect(rect, pad_px=pad_px, width=w, height=h)
		x, y, rw, rh = padded_rect

		roi = working_binary[y : y + rh, x : x + rw]
		if int(cv2.countNonZero(roi)) < int(min_white_pixels):
			break

		cropped = image[y : y + rh, x : x + rw]
		results.append((cropped, padded_rect))

		# Black out selected region so next iteration finds the next-best area.
		working_binary[y : y + rh, x : x + rw] = 0

	return results


def extract_clock_regions_by_colon(
	image: np.ndarray,
	processed_signal_image: np.ndarray,
	threshold: int = 127,
	min_area: int = 80,
	neighbor_distance_px: int = 24,
	pad_px: int = 8,
) -> list[tuple[np.ndarray, Rect]]:
	"""
	Extract clock groups in stable order using colon anchor.

	Flow:
	1) Find colon (two vertically aligned dots).
	2) Split candidate components by left/right of colon center.
	3) Grow each side recursively with high neighbor-awareness.
	4) Return crops in stable order: left group, then right group.
	"""
	if image is None:
		raise ValueError("image is None")
	if processed_signal_image is None:
		raise ValueError("processed_signal_image is required")

	binary = _to_binary_white_foreground(processed_signal_image, threshold=threshold)
	colon_rect = _find_colon_rect(binary)
	if colon_rect is None:
		return []

	colon_cx = _rect_center_x(colon_rect)
	all_rects = _extract_component_rects(binary, min_area=min_area)

	# Exclude the colon components and near-colon tiny fragments.
	filtered_rects: list[Rect] = []
	for rect in all_rects:
		if _rect_gap(rect, colon_rect) <= max(2, int(neighbor_distance_px // 2)):
			continue
		filtered_rects.append(rect)

	left_rects = [r for r in filtered_rects if _rect_center_x(r) < colon_cx]
	right_rects = [r for r in filtered_rects if _rect_center_x(r) > colon_cx]

	def _side_group(rects: list[Rect], side: str) -> Rect | None:
		if not rects:
			return None

		if side == "left":
			seed_idx = max(range(len(rects)), key=lambda i: _rect_center_x(rects[i]))
		else:
			seed_idx = min(range(len(rects)), key=lambda i: _rect_center_x(rects[i]))

		cluster = _grow_neighbor_cluster(seed_idx, rects, neighbor_distance_px=neighbor_distance_px)
		return _union_rects(cluster)

	left_group = _side_group(left_rects, "left")
	right_group = _side_group(right_rects, "right")

	h, w = image.shape[:2]
	results: list[tuple[np.ndarray, Rect]] = []
	for group in [left_group, right_group]:
		if group is None:
			continue
		padded = _pad_rect(group, pad_px=pad_px, width=w, height=h)
		x, y, rw, rh = padded
		results.append((image[y : y + rh, x : x + rw], padded))

	return results


def extract_full_clock_region_by_colon(
	image: np.ndarray,
	processed_signal_image: np.ndarray,
	threshold: int = 127,
	min_area: int = 80,
	neighbor_distance_px: int = 24,
	pad_px: int = 8,
) -> tuple[np.ndarray, Rect] | None:
	"""
	Extract one crop covering full time display: left digits + colon + right digits.

	Returns None when a reliable colon anchor cannot be found.
	"""
	if image is None:
		raise ValueError("image is None")
	if processed_signal_image is None:
		raise ValueError("processed_signal_image is required")

	binary = _to_binary_white_foreground(processed_signal_image, threshold=threshold)
	colon_rect = _find_colon_rect(binary)
	if colon_rect is None:
		return None

	h, w = image.shape[:2]
	colon_cx = _rect_center_x(colon_rect)
	colon_cy = colon_rect[1] + (colon_rect[3] / 2.0)

	candidate_rects = _extract_component_rects(binary, min_area=min_area)
	selected: list[Rect] = []

	vertical_band = max(20.0, colon_rect[3] * 2.2)
	horizontal_span = max(float(w) * 0.35, 260.0)

	for rect in candidate_rects:
		if _rect_gap(rect, colon_rect) <= 2:
			continue

		rcx = _rect_center_x(rect)
		rcy = rect[1] + (rect[3] / 2.0)
		if abs(rcy - colon_cy) > vertical_band:
			continue
		if abs(rcx - colon_cx) > horizontal_span:
			continue

		selected.append(rect)

	# Require at least one component on each side for stable full-time crop.
	left = [r for r in selected if _rect_center_x(r) < colon_cx]
	right = [r for r in selected if _rect_center_x(r) > colon_cx]

	if left and right:
		full_rect = _union_rects(selected + [colon_rect])
	else:
		# Fallback to previous left/right grouping behavior.
		clock_groups = extract_clock_regions_by_colon(
			image=image,
			processed_signal_image=processed_signal_image,
			threshold=threshold,
			min_area=min_area,
			neighbor_distance_px=neighbor_distance_px,
			pad_px=0,
		)
		if not clock_groups:
			return None
		group_rects = [rect for _, rect in clock_groups]
		full_rect = _union_rects(group_rects + [colon_rect])

	padded = _pad_rect(full_rect, pad_px=pad_px, width=w, height=h)
	x, y, rw, rh = padded

	return image[y : y + rh, x : x + rw], padded


def split_connected_digits_left_to_right(
	binary_digit_line: np.ndarray,
	min_digit_area: int = 80,
	min_digit_height: int = 20,
) -> list[tuple[np.ndarray, Rect]]:
	"""
	Split a binary clock crop into ordered digit crops using connected components.

	Assumptions:
	- each digit is a single connected component
	- digits are not connected to each other
	"""
	if binary_digit_line is None:
		raise ValueError("binary_digit_line is None")

	if len(binary_digit_line.shape) == 3:
		gray = cv2.cvtColor(binary_digit_line, cv2.COLOR_BGR2GRAY)
	else:
		gray = binary_digit_line

	_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	h, w = binary.shape[:2]

	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
	digits: list[tuple[np.ndarray, Rect]] = []

	for idx in range(1, num_labels):
		x = int(stats[idx, cv2.CC_STAT_LEFT])
		y = int(stats[idx, cv2.CC_STAT_TOP])
		rw = int(stats[idx, cv2.CC_STAT_WIDTH])
		rh = int(stats[idx, cv2.CC_STAT_HEIGHT])
		area = int(stats[idx, cv2.CC_STAT_AREA])

		if area < int(min_digit_area):
			continue
		if rh < int(min_digit_height):
			continue
		# Drop extremely tiny-width blobs that are usually punctuation noise.
		if rw <= 2:
			continue

		crop = binary[y : y + rh, x : x + rw]
		digits.append((crop, (x, y, rw, rh)))

	# Stable reading order.
	digits.sort(key=lambda item: item[1][0])
	return digits
