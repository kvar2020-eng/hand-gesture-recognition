"""Validate dataset folders and class mappings without inventing label meanings."""
import argparse
import json
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def validate_labels(labels):
    if not isinstance(labels, list) or len(labels) < 2:
        raise ValueError("At least two gesture classes are required.")
    if any(not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label)
           for label in labels):
        raise ValueError("Use class names containing letters, numbers, '_' or '-'.")
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError("Class names must be unique, including letter case.")
    return labels


def label_argument(value):
    if not LABEL_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "Label must start with a letter/number and use only letters, numbers, '_' or '-'.")
    return value


def split_counts(directory, expected_labels=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Missing dataset folder: {directory}")
    folders = sorted(path for path in directory.iterdir()
                     if path.is_dir() and not path.name.startswith("."))
    labels = validate_labels([path.name for path in folders])
    if expected_labels is not None and labels != list(expected_labels):
        raise ValueError(f"Class folders in {directory} do not match the training classes.")
    counts = {
        path.name: sum(1 for image in path.rglob("*")
                       if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS)
        for path in folders
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Every class in {directory} needs at least one image.")
    return counts


def read_labels(path):
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_labels(labels)
