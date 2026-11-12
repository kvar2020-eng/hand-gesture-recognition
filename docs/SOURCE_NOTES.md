# Source and reconstruction notes

## Source provided

The project was prepared from Python code fragments supplied by the user from a report titled `OJ036E123CT.pdf`. Those fragments described background subtraction, a fixed image ROI, a convolutional classifier, and webcam inference. The original complete Python files, dataset, class mapping, training history, and trained weights were not supplied.

The original report and Canva collaboration/access URLs are not included in this repository. The repository does not claim to be an exact recovery of the report's original implementation.

## What was preserved

- OpenCV webcam input and a horizontally mirrored view.
- Default ROI X=150, Y=100, width=200, height=200.
- Empty-background calibration using accumulated weighted averaging.
- Thresholded foreground masks and largest-contour selection.
- 64x64 three-channel CNN input.
- Three Conv2D stages with 32, 64, and 128 filters, max pooling, then dense layers with 64, 128, and 128 units.

## Repairs and new implementation

| Issue in the fragments | Treatment in this project |
| --- | --- |
| Broken imports, indentation, missing parentheses, and prose mixed into code | Reorganized into valid Python modules using `tf.keras` consistently |
| Incomplete `accumulateWeighted` and `threshold` calls | Completed calls in a reusable background segmenter |
| Old three-return-value contour unpacking | Compatible contour selection for the current OpenCV API |
| Missing collection-loop context and undefined counters/label variables | New explicit collection CLI with safe filenames and separate split folders |
| Hard-coded Windows drive paths | Portable `pathlib` paths and configurable dataset/model directories |
| Missing model compilation, training, and persistence | New training command, validation callbacks, and saved artifacts |
| Undefined `word_dict` and unknown ten-class label meanings | Ordered labels generated from actual training folders; no guessed alphabet or gesture mapping |
| Fixed ten-output classifier | Output dimension derived from the dataset's actual classes |
| Unspecified input normalization | Shared 64x64 resize convention and model-owned rescaling |
| No guard for `segment_hand(...)` returning `None` | Explicit no-foreground path and minimum contour-area check |
| Incomplete failure/cleanup paths | Camera-read validation, bounded ROI checks, and guaranteed resource cleanup |
| Inference from an untrained in-memory model | Require a saved model and matching labels before opening the camera |

Training, class persistence, CLI controls, input validation, minimum-area filtering, uncertain-score display, and tests are **reconstructed additions**. They were not recovered as complete source from the report. No training data, model weights, or recognition metrics were fabricated.

## Attribution context

One provided overlay string reads `DataFlair hand sign recognition`. This may indicate tutorial provenance; it is recorded as context, not as a verified claim that DataFlair authored every fragment. An upstream URL, exact version, license, report authorship, and each contributor's original work were not provided or independently verified.

Before treating this as a portfolio submission or redistributing source-derived material, add accurate contributor information and verify the original source's attribution and license requirements. No license was selected automatically. This reconstruction does not assert sole original authorship of the report's ideas or fragments.
