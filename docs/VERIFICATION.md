# Verification and current limits

- Python used for these checks: `3.13.18`.
- All supplied application and test files passed Python syntax compilation.
- The `train.py --help` command completed successfully without TensorFlow.
- Dataset/label-validation tests were run. Image-processing and camera CLI tests could not be run in this sandbox: its GUI OpenCV installation cannot import because `libGL.so.1` is missing, and a usable headless test installation was not available.

```text
test_at_least_two_classes_needed (test_dataset.DatasetTests.test_at_least_two_classes_needed) ... ok
test_case_conflicting_labels_are_rejected (test_dataset.DatasetTests.test_case_conflicting_labels_are_rejected) ... ok
test_empty_class_is_rejected (test_dataset.DatasetTests.test_empty_class_is_rejected) ... ok
test_invalid_labels_are_rejected (test_dataset.DatasetTests.test_invalid_labels_are_rejected) ... ok
test_json_mapping_must_be_a_list (test_dataset.DatasetTests.test_json_mapping_must_be_a_list) ... ok
test_json_mapping_preserves_class_order (test_dataset.DatasetTests.test_json_mapping_preserves_class_order) ... ok
test_labels_inferred_from_folder_names (test_dataset.DatasetTests.test_labels_inferred_from_folder_names) ... ok
test_missing_dataset_is_rejected (test_dataset.DatasetTests.test_missing_dataset_is_rejected) ... ok
test_model_rejects_one_class_before_tensorflow_import (test_dataset.DatasetTests.test_model_rejects_one_class_before_tensorflow_import) ... ok
test_optional_model_architecture (test_dataset.DatasetTests.test_optional_model_architecture) ... skipped 'TensorFlow is not installed'
test_validation_classes_must_match (test_dataset.DatasetTests.test_validation_classes_must_match) ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.004s

OK (skipped=1)
```

## Not verified

- TensorFlow is not installed in this sandbox; model construction, model fitting, checkpoints, and saved-model inference were not executed. The TensorFlow architecture test is skipped.
- No physical webcam, GUI session, training dataset, trained weights, or class mapping was supplied. No end-to-end recognition test or accuracy measurement was performed.
- Dependency version ranges are not a fully tested environment lockfile. Run the included checks again in your supported local environment before relying on the application.
- Core/data tests are not evidence of sign-language understanding or real-world classification performance.
