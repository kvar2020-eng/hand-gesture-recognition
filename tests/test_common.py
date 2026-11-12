import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from common import HandSegmenter, ROI, model_input, open_camera, read_frame


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.background = np.zeros((100, 100), dtype=np.uint8)

    def test_roi_crop(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(ROI().crop(frame).shape, (200, 200, 3))

    def test_roi_rejects_invalid_coordinates(self):
        for values in [(-1, 0, 20, 20), (0, -1, 20, 20), (0, 0, 0, 20)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ROI(*values)

    def test_roi_rejects_out_of_bounds(self):
        with self.assertRaises(ValueError):
            ROI().crop(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_requires_calibration(self):
        with self.assertRaises(RuntimeError):
            HandSegmenter().segment(self.background)

    def test_static_background_has_no_foreground(self):
        segmenter = HandSegmenter()
        segmenter.calibrate(self.background)
        self.assertIsNone(segmenter.segment(self.background))

    def test_detects_largest_foreground(self):
        segmenter = HandSegmenter(min_area=20)
        segmenter.calibrate(self.background)
        frame = self.background.copy()
        frame[20:60, 20:60] = 255
        frame[80:84, 80:84] = 255
        mask, contour = segmenter.segment(frame)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(cv2.contourArea(contour), 1000)

    def test_filters_tiny_contours(self):
        segmenter = HandSegmenter(min_area=500)
        segmenter.calibrate(self.background)
        frame = self.background.copy()
        frame[20:25, 20:25] = 255
        self.assertIsNone(segmenter.segment(frame))

    def test_background_accumulation(self):
        segmenter = HandSegmenter(weight=0.5)
        segmenter.calibrate(self.background)
        segmenter.calibrate(np.full_like(self.background, 100))
        self.assertTrue(np.allclose(segmenter.background, 50))

    def test_reset_requires_recalibration(self):
        segmenter = HandSegmenter()
        segmenter.calibrate(self.background)
        segmenter.reset()
        with self.assertRaises(RuntimeError):
            segmenter.segment(self.background)

    def test_changed_roi_size_is_rejected(self):
        segmenter = HandSegmenter()
        segmenter.calibrate(self.background)
        for operation in [segmenter.calibrate, segmenter.segment]:
            with self.subTest(operation=operation.__name__), self.assertRaises(ValueError):
                operation(np.zeros((20, 20), dtype=np.uint8))

    def test_invalid_parameters(self):
        for values in [dict(weight=0), dict(weight=1.1), dict(threshold=-1),
                       dict(threshold=256), dict(min_area=-1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                HandSegmenter(**values)

    def test_invalid_grayscale_input(self):
        for image in [np.zeros((0, 0), dtype=np.uint8),
                      np.zeros((4, 4, 3), dtype=np.uint8),
                      np.zeros((4, 4), dtype=np.float32)]:
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                model_input(image)

    def test_model_input_matches_training_pixel_range(self):
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = model_input(mask)
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 255)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    @patch('common.cv2.destroyAllWindows')
    @patch('common.cv2.VideoCapture')
    def test_camera_released_after_loop_error(self, capture, destroy):
        camera = Mock()
        camera.isOpened.return_value = True
        capture.return_value = camera
        with self.assertRaisesRegex(RuntimeError, 'loop failed'):
            with open_camera(0):
                raise RuntimeError('loop failed')
        camera.release.assert_called_once()
        destroy.assert_called_once()

    @patch('common.cv2.destroyAllWindows')
    @patch('common.cv2.VideoCapture')
    def test_unavailable_camera_is_closed(self, capture, destroy):
        camera = Mock()
        camera.isOpened.return_value = False
        capture.return_value = camera
        with self.assertRaisesRegex(RuntimeError, 'Cannot open camera'):
            with open_camera(0):
                self.fail('An unavailable camera should never be yielded')
        camera.release.assert_called_once()

    def test_failed_camera_read_is_not_processed(self):
        camera = Mock()
        camera.read.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, 'stopped returning'):
            read_frame(camera, ROI())


if __name__ == '__main__':
    unittest.main()
