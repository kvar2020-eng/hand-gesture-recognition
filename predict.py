"""Run an already-trained gesture classifier on a locally connected webcam."""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from common import (HandSegmenter, ROI, camera_arguments, model_input,
                    open_camera, overlay, read_frame)
from dataset import read_labels


def confidence_argument(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Confidence threshold must be between 0 and 1.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/run1"))
    parser.add_argument("--confidence", type=confidence_argument, default=0.7)
    camera_arguments(parser)
    args = parser.parse_args()

    try:
        roi = ROI(*args.roi)
        segmenter = HandSegmenter(threshold=args.threshold, min_area=args.min_area)
        model_path = args.model_dir / "model.keras"
        labels_path = args.model_dir / "labels.json"
        if not model_path.is_file() or not labels_path.is_file():
            raise ValueError("No saved model and label mapping found; collect data and run train.py first.")
        labels = read_labels(labels_path)
        import tensorflow as tf
        # Load only model artifacts you trust; .keras safe mode remains enabled.
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
        if model.input_shape != (None, 64, 64, 3):
            raise ValueError("Model must accept 64x64 RGB images.")
        if model.output_shape != (None, len(labels)):
            raise ValueError("Saved labels do not match the model's output classes.")
        frames = 0
        print("Keep the box empty while calibrating. R: recalibrate; Q/Esc: quit.")
        with open_camera(args.camera) as camera:
            while True:
                frame, gray = read_frame(camera, roi)
                contour = None
                mask = np.zeros_like(gray)
                if frames < args.warmup_frames:
                    segmenter.calibrate(gray)
                    status = "Calibrating: keep hand outside the box"
                else:
                    result = segmenter.segment(gray)
                    status = "No foreground detected"
                    if result is not None:
                        mask, contour = result
                        scores = np.asarray(model(model_input(mask)[None, ...], training=False))[0]
                        if not np.all(np.isfinite(scores)):
                            raise RuntimeError("Model returned invalid prediction values.")
                        index = int(np.argmax(scores))
                        score = float(scores[index])
                        label = labels[index] if score >= args.confidence else "Uncertain"
                        status = f"{label}: {score:.1%} model score (not calibrated)"
                overlay(frame, roi, [status, "R: reset background | Q/Esc: quit"], contour)
                cv2.imshow("Thresholded ROI", mask)
                cv2.imshow("Gesture recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    segmenter.reset()
                    frames = 0
                    continue
                frames += 1
    except ImportError:
        parser.exit(2, "TensorFlow is required for inference. Install requirements.txt in a supported Python environment.\n")
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
