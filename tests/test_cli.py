import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def run_command(self, script, *arguments):
        return subprocess.run([sys.executable, str(ROOT / script), *arguments],
                              cwd=ROOT, text=True, capture_output=True, timeout=20)

    def test_all_help_commands(self):
        for script in ['collect.py', 'train.py', 'predict.py']:
            with self.subTest(script=script):
                result = self.run_command(script, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_invalid_label_is_rejected_before_camera_access(self):
        result = self.run_command('collect.py', '--label', '../escape', '--split', 'train')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Label must start', result.stderr)

    def test_missing_model_is_rejected_before_camera_or_tensorflow(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_command('predict.py', '--model-dir', directory)
        self.assertEqual(result.returncode, 2)
        self.assertIn('run train.py first', result.stderr)

    def test_missing_data_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_command('train.py', '--dataset', directory)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Missing dataset folder', result.stderr)


if __name__ == '__main__':
    unittest.main()
