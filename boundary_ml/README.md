# boundary_ml - Digit Presence Detector

This folder contains a one-class object detection pipeline.
The detector only answers: does this box contain a digit?

## Goal

- Detect and localize digits with one class: `digit`
- Do not classify identities (`0` through `9`)
- CPU-first defaults for Windows

## Folder Contract

- `src/` scripts for generation, training, evaluation, prediction
- `data/synth/` synthetic detection dataset (generated)
- `data/real/` manually labeled real images (you add)
- `data/hybrid/` merged synthetic + real training set
- `models/` exported model weights
- `runs/` YOLO run outputs and reports

## Dependencies

Use the same core stack already present in `programs/scoreboard_site/requirements.txt`:

- ultralytics
- opencv-python-headless
- Pillow
- pandas

Install from repo root:

`pip install -r programs/scoreboard_site/requirements.txt`

## 1) Generate Synthetic Detection Data

From repo root:

`python boundary_ml/src/generate_synth_dataset.py --train-count 2400 --val-count 400 --negative-ratio 0.2`

What it does:
- Reads cropped digits from `programs/data/data/0..9`
- Composes random scenes
- Writes YOLO labels with one class id `0`
- Creates `boundary_ml/data/synth/data.yaml`

Optional label sanity check:

`python boundary_ml/src/inspect_labels.py --data-root boundary_ml/data/synth --split train --count 20`

## 2) Train Baseline on Synthetic Data

`python boundary_ml/src/train_synth.py --data boundary_ml/data/synth/data.yaml --device cpu`

Best model is copied to:

`boundary_ml/models/digit_detector_synth.pt`

## 3) Prepare Real Labeled Data for Fine-Tuning

Create this structure and add matching image/label pairs:

- `boundary_ml/data/real/images/train/*.jpg|png`
- `boundary_ml/data/real/labels/train/*.txt`
- `boundary_ml/data/real/images/val/*.jpg|png`
- `boundary_ml/data/real/labels/val/*.txt`

YOLO label format per line:

`0 x_center y_center width height`

All coordinates must be normalized to `[0, 1]`.

## 4) Merge Synthetic + Real

`python boundary_ml/src/merge_datasets.py --synth-root boundary_ml/data/synth --real-root boundary_ml/data/real --out-root boundary_ml/data/hybrid`

This creates:

- `boundary_ml/data/hybrid/images/train`
- `boundary_ml/data/hybrid/labels/train`
- `boundary_ml/data/hybrid/images/val`
- `boundary_ml/data/hybrid/labels/val`
- `boundary_ml/data/hybrid/data.yaml`

## 5) Fine-Tune on Hybrid Data

`python boundary_ml/src/finetune_hybrid.py --data boundary_ml/data/hybrid/data.yaml --weights boundary_ml/models/digit_detector_synth.pt --device cpu`

Best hybrid model is copied to:

`boundary_ml/models/digit_detector_hybrid.pt`

## 6) Evaluate

`python boundary_ml/src/evaluate_detector.py --weights boundary_ml/models/digit_detector_hybrid.pt --data boundary_ml/data/hybrid/data.yaml --device cpu`

Metrics JSON is saved to:

`boundary_ml/runs/eval/metrics.json`

## 7) Predict Bounding Boxes

`python boundary_ml/src/predict.py --weights boundary_ml/models/digit_detector_hybrid.pt --source boundary_ml/data/hybrid/images/val --conf 0.20 --device cpu`

Outputs:
- Annotated prediction images in `boundary_ml/runs/predict*`
- Plain boxes JSON in `boundary_ml/runs/predict/boxes.json`

## Notes

- Start with synthetic data to get a working detector quickly.
- Then fine-tune on real labeled examples to reduce domain gap.
- For CPU-only runs on Windows, keep:
  - `imgsz` in the 416 to 640 range
  - small `batch` (2 to 4)
  - `workers=0`
