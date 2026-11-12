"""Record thresholded ROI images into explicitly separate dataset splits."""
import argparse
from pathlib import Path
from uuid import uuid4

import cv2

from common import (HandSegmenter, ROI, camera_arguments, open_camera,
                    overlay, positive_int, read_frame)
from dataset import label_argument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, type=label_argument)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--dataset", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=positive_int, default=300)
    parser.add_argument("--every", type=positive_int, default=5,
                        help="Save at most one image every N frames while recording.")
    camera_arguments(parser)
    args = parser.parse_args()

    try:
        roi = ROI(*args.roi)
        segmenter = HandSegmenter(threshold=args.threshold, min_area=args.min_area)
        output = args.dataset / args.split / args.label
        output.mkdir(parents=True, exist_ok=True)
        saved, frames, recording = 0, 0, False
        print("Keep the ROI empty during calibration. Space: record/pause; R: recalibrate; Q/Esc: quit.")
        with open_camera(args.camera) as camera:
            while saved < args.samples:
                frame, gray = read_frame(camera, roi)
                contour = None
                if frames < args.warmup_frames:
                    segmenter.calibrate(gray)
                    status = "Calibrating: keep hand outside the box"
                else:
                    result = segmenter.segment(gray)
                    status = "Paused - Space to record" if not recording else "Recording"
                    if result is None:
                        status = "No foreground detected"
                    else:
                        mask, contour = result
                        cv2.imshow("Thresholded ROI", mask)
                        if recording and frames % args.every == 0:
                            filename = output / f"{uuid4().hex}.png"
                            if not cv2.imwrite(str(filename), mask):
                                raise OSError(f"Could not save image: {filename}")
                            saved += 1
                overlay(frame, roi, [f"{args.label}: {saved}/{args.samples}", status,
                                    "Space: record/pause | R: reset | Q/Esc: quit"], contour)
                cv2.imshow("Collect gesture data", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    segmenter.reset()
                    frames, recording = 0, False
                    continue
                if key == ord(" ") and frames >= args.warmup_frames:
                    recording = not recording
                frames += 1
        print(f"Saved {saved} images to {output}")
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
