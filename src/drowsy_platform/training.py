from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from drowsy_platform.config import TrainingSettings


def train_and_export(settings: TrainingSettings) -> Path:
    import tensorflow as tf

    settings.output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = settings.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        settings.dataset_path,
        validation_split=settings.validation_split,
        subset="training",
        seed=settings.random_seed,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        settings.dataset_path,
        validation_split=settings.validation_split,
        subset="validation",
        seed=settings.random_seed,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(autotune)

    model = _build_model(tf, settings)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
        )
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=settings.epochs, callbacks=callbacks)
    loss, accuracy = model.evaluate(val_ds, verbose=0)

    model_path = run_dir / "model.keras"
    model.save(model_path)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(settings.dataset_path),
        "model_type": settings.model_type,
        "class_names": class_names,
        "image_size": list(settings.image_size),
        "epochs_requested": settings.epochs,
        "validation_accuracy": float(accuracy),
        "validation_loss": float(loss),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "model_path": str(model_path),
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _build_model(tf, settings: TrainingSettings):
    if settings.model_type == "simple":
        return _build_simple_cnn(tf, settings)
    if settings.model_type == "mobilenet":
        return _build_mobilenet(tf, settings)
    raise ValueError("TRAIN_MODEL_TYPE must be 'simple' or 'mobilenet'.")


def _build_simple_cnn(tf, settings: TrainingSettings):
    inputs = tf.keras.Input(shape=(settings.image_height, settings.image_width, 3))
    x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _build_mobilenet(tf, settings: TrainingSettings):
    base_model = tf.keras.applications.MobileNetV2(
        weights=settings.base_weights,
        include_top=False,
        input_shape=(settings.image_height, settings.image_width, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(settings.image_height, settings.image_width, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def promote_candidate(
    candidate_manifest_path: Path,
    production_dir: Path,
    min_accuracy: float,
    min_delta: float,
) -> bool:
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_accuracy = float(candidate_manifest["validation_accuracy"])

    if candidate_accuracy < min_accuracy:
        return False

    production_manifest_path = production_dir / "manifest.json"
    current_accuracy = 0.0
    if production_manifest_path.exists():
        production_manifest = json.loads(production_manifest_path.read_text(encoding="utf-8"))
        current_accuracy = float(production_manifest.get("validation_accuracy", 0.0))

    if candidate_accuracy < current_accuracy + min_delta:
        return False

    production_dir.mkdir(parents=True, exist_ok=True)
    candidate_model_path = Path(candidate_manifest["model_path"])
    production_model_path = production_dir / candidate_model_path.name
    promoted_manifest = dict(candidate_manifest)
    promoted_manifest["model_path"] = str(production_model_path)
    production_manifest_path.write_text(json.dumps(promoted_manifest, indent=2), encoding="utf-8")
    production_model_path.write_bytes(candidate_model_path.read_bytes())
    return True
