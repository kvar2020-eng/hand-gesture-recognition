# Hand Gesture Recognition

An educational Python project for collecting and classifying isolated, static hand gestures with OpenCV and a TensorFlow/Keras CNN.

> **Project status:** Refactored and partly reconstructed from supplied report snippets, not an exact recovery of the original source. No dataset, pretrained model, or measured recognition accuracy is included. This is **not** a complete sign-language translator or a production accessibility tool.

## What is included

- Webcam collection of thresholded region-of-interest (ROI) images.
- Background calibration and largest-foreground-contour segmentation.
- A training command based on the report's three-convolution CNN.
- Automatic class mapping from your dataset folders, saved with the model.
- Live prediction using a saved model, with a configurable uncertain-score threshold.
- Unit tests for core image processing, dataset validation, and command-line safeguards.

See [source and reconstruction notes](docs/SOURCE_NOTES.md) and [verification results](docs/VERIFICATION.md).

## Requirements

- Python **3.11 is recommended**; check TensorFlow wheel support for your OS/Python version.
- A local computer with a webcam, camera permission, and a graphical desktop.
- A fixed camera and reasonably stable lighting/background.

GitHub hosts the source code. The webcam application runs on your computer, not on your GitHub profile or GitHub Pages.

## Setup

Download and extract the project ZIP, then open a terminal in the extracted project folder:

```bash
cd hand-gesture-recognition
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows Command Prompt
.venv\Scripts\activate
```

Install the application dependencies:

```bash
python -m pip install -r requirements.txt
```

The version ranges are compatibility constraints, not a fully tested lockfile. Use a supported TensorFlow environment and record resolved package versions when reporting experiments. Linux desktop installations may also need system graphics libraries for OpenCV GUI windows.

## 1. Collect data

Folder names define your classes. `fist` and `open_palm` below are **examples**, not a supplied or pretrained label set. Use at least two classes and the same class names in each split.

```text
data/
  train/
    fist/
    open_palm/
  validation/
    fist/
    open_palm/
  test/                 # optional independent evaluation data
    fist/
    open_palm/
```

Collect each training class:

```bash
python collect.py --label fist --split train --samples 300
python collect.py --label open_palm --split train --samples 300
```

Collect validation images in **separate recording sessions**, not by copying or randomly splitting adjacent frames from the same session:

```bash
python collect.py --label fist --split validation --samples 75
python collect.py --label open_palm --split validation --samples 75
```

Controls:

- Keep your hand **outside the box** during the first 70 calibration frames.
- Put the intended gesture inside the box after calibration.
- **Space:** start/pause saving. Only thresholded ROI images are saved.
- **R:** reset the background and recalibrate with an empty box.
- **Q** or **Esc:** quit; camera resources are released.

Camera options are shared by collection and prediction:

```bash
python collect.py --label fist --split train --camera 0 --roi 150 100 200 200
```

`--roi` means `X Y WIDTH HEIGHT` in the horizontally mirrored camera frame. The ROI must fit inside that frame. Use consistent ROI and segmentation settings during data collection and inference. See `--help` for the threshold, contour-area, frame-spacing, and calibration options.

Avoid recording sensitive surroundings. Foreground segmentation is not a learned hand detector: moving objects and changing lighting can trigger it too.

## 2. Train your model

```bash
python train.py --dataset data --output models/run1 --epochs 20
```

Training:

1. Validates matching, nonempty class folders in `train` and `validation`.
2. Loads 64x64 RGB images with the same nearest-neighbor resizing used for inference.
3. Normalizes pixels inside the model, then trains the CNN.
4. Saves the best-validation-loss `model.keras`, its ordered `labels.json`, and actual run history in `training.json`.

The classifier output count comes from your folders, rather than assuming the report's ten outputs have known meanings. A nonempty output directory is rejected to avoid overwriting an earlier run; choose a new `--output` directory when retraining.

Validation loss is used for model selection. Do not present validation scores as independent test accuracy. For any published performance claim, use a separate test set with independently recorded sessions/participants, inspect per-class errors, and document the evaluation method. This repository does not include an evaluation report or accuracy claim.

## 3. Predict locally

```bash
python predict.py --model-dir models/run1 --confidence 0.7
```

A trained model **and its matching label file** are required. There is no fallback that presents a randomly initialized network as a working classifier. Load only model artifacts you trust.

Keep the box empty during calibration, then place a gesture inside it. Use **R** to recalibrate and **Q/Esc** to exit. Low scores display `Uncertain`. A softmax score is not calibrated confidence; unknown gestures can still receive high scores.

## Run checks

In the application environment:

```bash
python -m compileall -q common.py dataset.py collect.py train.py predict.py tests
python -m unittest discover -s tests -v
```

For core tests without a desktop or TensorFlow, create a **separate** environment and install `requirements-test.txt` instead of `requirements.txt`. Do not install both the headless and GUI OpenCV distributions in the same environment. The model-architecture test is skipped when TensorFlow is absent; skipping it is not evidence that model training was tested.

## Limitations and responsible use

- Background subtraction is sensitive to lighting, camera movement, sleeves, and unrelated objects.
- The CNN must be trained on representative labeled examples; performance is not established here.
- Consecutive webcam frames are highly correlated. Separate sessions and people across training, validation, and testing to reduce leakage.
- Static isolated gestures do not cover sign-language grammar, motion, facial expression, or complete language understanding.
- Do not rely on this prototype for medical, emergency, or accessibility-critical communication.
- Datasets, weights, recordings, and environment files are ignored by Git. Review every file before any upload; `.gitignore` alone is not a privacy guarantee.

## Attribution and license

The supplied code fragments contain a `DataFlair` overlay string. Upstream provenance and licensing were not independently verified; see [source notes](docs/SOURCE_NOTES.md). No license has been selected on the author's behalf. Confirm attribution and permissions for source-derived material before redistribution or choosing a license.
