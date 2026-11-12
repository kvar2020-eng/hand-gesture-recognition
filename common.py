"""Shared preprocessing and camera helpers; no TensorFlow import required."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SIZE = (64, 64)


@dataclass(frozen=True)
class ROI:
    x: int = 150
    y: int = 100
    width: int = 200
    height: int = 200

    def __post_init__(self):
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("ROI needs nonnegative X/Y and positive width/height.")

    def crop(self, frame):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel camera frame.")
        height, width = frame.shape[:2]
        if self.x + self.width > width or self.y + self.height > height:
            raise ValueError("ROI is outside the camera frame; adjust --roi X Y W H.")
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]


def validate_gray(gray):
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.size == 0:
        raise ValueError("Expected a nonempty grayscale image.")
    if gray.dtype != np.uint8:
        raise ValueError("Grayscale images must use uint8 pixels.")


class HandSegmenter:
    """Background subtraction for a fixed camera, not a learned hand detector."""

    def __init__(self, weight=0.5, threshold=25, min_area=500):
        if not 0 < weight <= 1 or not 0 <= threshold <= 255 or min_area < 0:
            raise ValueError("Invalid background weight, threshold, or contour area.")
        self.weight = weight
        self.threshold = threshold
        self.min_area = min_area
        self.background = None

    def reset(self):
        self.background = None

    def calibrate(self, gray):
        validate_gray(gray)
        if self.background is None:
            self.background = gray.astype(np.float32)
        elif self.background.shape != gray.shape:
            raise ValueError("ROI size changed; reset background calibration.")
        else:
            cv2.accumulateWeighted(gray, self.background, self.weight)

    def segment(self, gray):
        validate_gray(gray)
        if self.background is None:
            raise RuntimeError("Calibrate the empty ROI before segmentation.")
        if self.background.shape != gray.shape:
            raise ValueError("ROI size changed; reset background calibration.")
        difference = cv2.absdiff(self.background.astype(np.uint8), gray)
        _, mask = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
        # OpenCV 4 returns (contours, hierarchy); [-2] also accepts OpenCV 3.
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return None
        return mask, contour


def model_input(mask):
    """64x64 RGB float32, 0..255; the model owns the 1/255 rescaling."""
    validate_gray(mask)
    resized = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32)


@contextmanager
def open_camera(index):
    camera = cv2.VideoCapture(index)
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open camera; check permission and --camera index.")
        yield camera
    finally:
        camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Cleanup must not hide an earlier error on a headless build.


def read_frame(camera, roi):
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError("Camera stopped returning frames.")
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(roi.crop(frame), cv2.COLOR_BGR2GRAY)
    return frame, cv2.GaussianBlur(gray, (9, 9), 0)


def overlay(frame, roi, lines, contour=None):
    cv2.rectangle(frame, (roi.x, roi.y),
                  (roi.x + roi.width, roi.y + roi.height), (255, 128, 0), 2)
    if contour is not None:
        offset = np.array([roi.x, roi.y], dtype=contour.dtype)
        cv2.drawContours(frame, [contour + offset], -1, (0, 255, 0), 1)
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (10, 24 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1.")
    return number


def camera_arguments(parser):
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi", nargs=4, type=int, default=[150, 100, 200, 200],
                        metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--warmup-frames", type=positive_int, default=70)
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--min-area", type=int, default=500)
