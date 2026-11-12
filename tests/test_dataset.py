import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import struct
import zlib

from dataset import read_labels, split_counts, validate_labels
from train import build_model


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def create_split(self, split, labels):
        for label in labels:
            directory = self.root / split / label
            directory.mkdir(parents=True)
            def chunk(kind, data):
                return (struct.pack('>I', len(data)) + kind + data
                        + struct.pack('>I', zlib.crc32(kind + data)))
            image = (b'\x89PNG\r\n\x1a\n'
                     + chunk(b'IHDR', struct.pack('>IIBBBBB', 8, 8, 8, 0, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress((b'\x00' * 9) * 8))
                     + chunk(b'IEND', b''))
            (directory / 'sample.png').write_bytes(image)
        return self.root / split

    def test_labels_inferred_from_folder_names(self):
        path = self.create_split('train', ['open_palm', 'fist'])
        self.assertEqual(split_counts(path), {'fist': 1, 'open_palm': 1})

    def test_validation_classes_must_match(self):
        path = self.create_split('validation', ['fist', 'peace'])
        with self.assertRaisesRegex(ValueError, 'do not match'):
            split_counts(path, ['fist', 'open_palm'])

    def test_missing_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing dataset'):
            split_counts(self.root / 'missing')

    def test_empty_class_is_rejected(self):
        path = self.create_split('train', ['fist', 'open_palm'])
        (path / 'fist' / 'sample.png').unlink()
        with self.assertRaisesRegex(ValueError, 'at least one image'):
            split_counts(path)

    def test_at_least_two_classes_needed(self):
        path = self.create_split('train', ['fist'])
        with self.assertRaisesRegex(ValueError, 'two gesture classes'):
            split_counts(path)

    def test_case_conflicting_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_labels(['A', 'a'])

    def test_invalid_labels_are_rejected(self):
        for labels in [['../secret', 'fist'], ['open palm', 'fist'], [1, 'fist']]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                validate_labels(labels)

    def test_json_mapping_preserves_class_order(self):
        path = self.root / 'labels.json'
        path.write_text(json.dumps(['open_palm', 'fist']))
        self.assertEqual(read_labels(path), ['open_palm', 'fist'])

    def test_json_mapping_must_be_a_list(self):
        path = self.root / 'labels.json'
        path.write_text(json.dumps({'0': 'fist', '1': 'open_palm'}))
        with self.assertRaises(ValueError):
            read_labels(path)

    def test_model_rejects_one_class_before_tensorflow_import(self):
        with self.assertRaisesRegex(ValueError, 'two classes'):
            build_model(1)

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is not installed')
    def test_optional_model_architecture(self):
        import numpy as np
        model = build_model(3)
        self.assertEqual(model.input_shape, (None, 64, 64, 3))
        self.assertEqual(model.output_shape, (None, 3))
        scores = np.asarray(model(np.zeros((1, 64, 64, 3)), training=False))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores.sum(axis=1), 1, atol=0.00001)


if __name__ == '__main__':
    unittest.main()
