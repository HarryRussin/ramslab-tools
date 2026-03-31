import random
from pathlib import Path

import cv2
from bound_brightest_spots import (
	extract_full_clock_region_by_colon,
	extract_top_signal_regions,
	split_connected_digits_left_to_right,
)
from filters import inverse_filter
from segment_digit_recognition import recognize_and_format


# For THRESH_BINARY_INV in this pipeline, a higher value usually keeps more pixels white.
PREPROCESS_THRESHOLD = 215
SIGNAL_THRESHOLD = 127
MIN_SIGNAL_AREA = 50
NEIGHBOR_MERGE_DISTANCE_PX = 65
CROP_PADDING = 28
MAX_REGIONS = 4


def preprocess_for_ocr(image, threshold=PREPROCESS_THRESHOLD):
	"""Preprocess pipeline: inverse -> binary threshold inverse."""
	inverted = inverse_filter(image)
	if len(inverted.shape) == 3:
		inverted = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
	_, processed = cv2.threshold(inverted, threshold, 255, cv2.THRESH_BINARY_INV)
	return processed


def run(input_path, output_path, output_dir):
	image = cv2.imread(str(input_path))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {input_path}")

	processed = preprocess_for_ocr(image, threshold=PREPROCESS_THRESHOLD)
	full_clock = extract_full_clock_region_by_colon(
		processed,
		processed_signal_image=processed,
		threshold=SIGNAL_THRESHOLD,
		min_area=MIN_SIGNAL_AREA,
		neighbor_distance_px=NEIGHBOR_MERGE_DISTANCE_PX,
		pad_px=CROP_PADDING,
	)

	if full_clock is not None:
		cropped_processed, rect = full_clock
		cv2.imwrite(str(output_path), cropped_processed)
		full_crop = cropped_processed
	else:
		# Fallback: union top regions into one box when colon anchor fails.
		regions = extract_top_signal_regions(
			processed,
			processed_signal_image=processed,
			threshold=SIGNAL_THRESHOLD,
			min_area=MIN_SIGNAL_AREA,
			neighbor_distance_px=NEIGHBOR_MERGE_DISTANCE_PX,
			pad_px=CROP_PADDING,
			max_regions=MAX_REGIONS,
		)

		if not regions:
			return None, [], [], ""

		rects = [rect for _, rect in regions]
		x1 = min(r[0] for r in rects)
		y1 = min(r[1] for r in rects)
		x2 = max(r[0] + r[2] for r in rects)
		y2 = max(r[1] + r[3] for r in rects)
		merged_rect = (x1, y1, x2 - x1, y2 - y1)
		x, y, w, h = merged_rect
		full_crop = processed[y : y + h, x : x + w]
		rect = merged_rect
		cv2.imwrite(str(output_path), full_crop)

	# Split full clock crop into ordered connected digits.
	digits = split_connected_digits_left_to_right(
		full_crop,
		min_digit_area=50,
		min_digit_height=20,
	)

	for i, (digit_img, _) in enumerate(digits, start=1):
		digit_path = output_dir / f"digit_{i}.png"
		cv2.imwrite(str(digit_path), digit_img)

	digit_images = [digit_img for digit_img, _ in digits]
	recognized_digits, pretty_time = recognize_and_format(digit_images)

	return rect, [digit_rect for _, digit_rect in digits], recognized_digits, pretty_time


def pick_random_image(dataset_dir):
	images = sorted(dataset_dir.glob("*.jpg")) + sorted(dataset_dir.glob("*.jpeg")) + sorted(dataset_dir.glob("*.png"))
	if not images:
		raise FileNotFoundError(f"No images found in: {dataset_dir}")
	return random.choice(images)


def main():
	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parent.parent
	dataset_dir = repo_root / "data" / "scoreboard_imgs"
	output_path = script_dir / "processed_random.png"

	input_path = pick_random_image(dataset_dir)
	rect, digit_rects, recognized_digits, pretty_time = run(input_path, output_path, script_dir)
	print(f"Input: {input_path}")
	if rect is not None:
		print(f"Output: {output_path}")
		print(f"Crop rect: {rect}")
		print(f"Digits found (left->right): {len(digit_rects)}")
		for i, drect in enumerate(digit_rects, start=1):
			print(f"Digit {i}: {script_dir / f'digit_{i}.png'} rect={drect}")
		print(f"Recognized digits: {' '.join(recognized_digits)}")
		print(f"Time (pretty): {pretty_time}")
	else:
		print("No signal regions found.")
	print(f"Threshold: {PREPROCESS_THRESHOLD}")
	print(f"Neighbor merge distance: {NEIGHBOR_MERGE_DISTANCE_PX}")


if __name__ == "__main__":
	main()
