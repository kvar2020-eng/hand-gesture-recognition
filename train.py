"""Train the report's CNN architecture on your own separated gesture datasets."""
import argparse
import json
from pathlib import Path

from dataset import split_counts


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1.")
    return number


def build_model(num_classes):
    if num_classes < 2:
        raise ValueError("At least two classes are required.")
    import tensorflow as tf
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="gesture_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/run1"))
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        training_counts = split_counts(args.dataset / "train")
        labels = list(training_counts)
        validation_counts = split_counts(args.dataset / "validation", labels)
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output is not an empty folder; choose a new --output path.")
        import tensorflow as tf
        tf.keras.utils.set_random_seed(args.seed)
        dataset_options = dict(labels="inferred", label_mode="int", class_names=labels,
                               color_mode="rgb", image_size=(64, 64),
                               interpolation="nearest", batch_size=args.batch_size)
        training = tf.keras.utils.image_dataset_from_directory(
            args.dataset / "train", shuffle=True, seed=args.seed, **dataset_options)
        validation = tf.keras.utils.image_dataset_from_directory(
            args.dataset / "validation", shuffle=False, **dataset_options)
        training = training.prefetch(tf.data.AUTOTUNE)
        validation = validation.prefetch(tf.data.AUTOTUNE)
        args.output.mkdir(parents=True, exist_ok=True)
        model = build_model(len(labels))
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(args.output / "model.keras"),
                                               monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                patience=2, min_lr=0.000001),
        ]
        history = model.fit(training, validation_data=validation, epochs=args.epochs,
                            callbacks=callbacks)
        (args.output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n",
                                                encoding="utf-8")
        record = {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "training_images": training_counts,
            "validation_images": validation_counts,
            "tensorflow_version": tf.__version__,
            "input_size": [64, 64, 3],
            "history": {key: [float(value) for value in values]
                        for key, values in history.history.items()},
        }
        (args.output / "training.json").write_text(json.dumps(record, indent=2) + "\n",
                                                  encoding="utf-8")
        print(f"Saved the best-validation-loss model and class mapping in {args.output}")
        print("Validation results are not held-out test accuracy or proof of generalization.")
    except ImportError:
        parser.exit(2, "TensorFlow is required for training. Install requirements.txt in a supported Python environment.\n")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
