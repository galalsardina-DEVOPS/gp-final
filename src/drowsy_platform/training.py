from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    class_counts = _count_classes(train_ds, class_names)
    class_weight = _compute_class_weight(class_counts) if settings.class_weight_enabled else None

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(autotune)

    model, build_metadata = _build_model(tf, settings)
    history_parts: list[Any] = []

    _compile_model(tf, model, settings.learning_rate)

    callbacks = _build_callbacks(tf, settings)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    history_parts.append(history)

    fine_tune_started = False
    if settings.model_type == "mobilenet" and settings.fine_tune_epochs > 0:
        fine_tune_started = _prepare_mobilenet_fine_tuning(tf, model, settings)
        if fine_tune_started:
            _compile_model(tf, model, settings.fine_tune_learning_rate)
            fine_tune_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=settings.fine_tune_epochs,
                callbacks=_build_callbacks(tf, settings),
                class_weight=class_weight,
            )
            history_parts.append(fine_tune_history)

    eval_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    accuracy = float(eval_metrics.get("accuracy", 0.0))
    loss = float(eval_metrics.get("loss", 0.0))

    model_path = run_dir / "model.keras"
    model.save(model_path)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(settings.dataset_path),
        "model_type": settings.model_type,
        "base_weights_requested": settings.base_weights,
        "base_weights_loaded": build_metadata.get("base_weights_loaded"),
        "class_names": class_names,
        "class_counts": class_counts,
        "class_weight": {str(key): float(value) for key, value in (class_weight or {}).items()},
        "image_size": list(settings.image_size),
        "epochs_requested": settings.epochs,
        "fine_tune_epochs_requested": settings.fine_tune_epochs,
        "fine_tune_started": fine_tune_started,
        "augmentation_enabled": settings.augmentation_enabled,
        "learning_rate": settings.learning_rate,
        "fine_tune_learning_rate": settings.fine_tune_learning_rate,
        "validation_accuracy": float(accuracy),
        "validation_loss": float(loss),
        "validation_metrics": {key: float(value) for key, value in eval_metrics.items()},
        "history": _merge_histories(history_parts),
        "model_path": str(model_path),
    }
    if build_metadata.get("base_weights_error"):
        manifest["base_weights_error"] = build_metadata["base_weights_error"]

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _build_model(tf, settings: TrainingSettings):
    if settings.model_type == "simple":
        return _build_simple_cnn(tf, settings), {"base_weights_loaded": None}
    if settings.model_type == "mobilenet":
        return _build_mobilenet(tf, settings)
    raise ValueError("TRAIN_MODEL_TYPE must be 'simple' or 'mobilenet'.")


def _compile_model(tf, model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def _build_callbacks(tf, settings: TrainingSettings) -> list[Any]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=settings.early_stopping_patience,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def _augmentation_layers(tf, settings: TrainingSettings):
    if not settings.augmentation_enabled:
        return tf.keras.layers.Activation("linear", name="augmentation")
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def _build_simple_cnn(tf, settings: TrainingSettings):
    inputs = tf.keras.Input(shape=(settings.image_height, settings.image_width, 3))
    x = _augmentation_layers(tf, settings)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu")(x)
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
    metadata: dict[str, str | None] = {
        "base_weights_loaded": settings.base_weights,
        "base_weights_error": None,
    }
    try:
        base_model = tf.keras.applications.MobileNetV2(
            weights=settings.base_weights,
            include_top=False,
            input_shape=(settings.image_height, settings.image_width, 3),
        )
    except Exception as exc:
        if settings.base_weights is None:
            raise
        metadata["base_weights_loaded"] = None
        metadata["base_weights_error"] = str(exc)
        base_model = tf.keras.applications.MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(settings.image_height, settings.image_width, 3),
        )
    base_model._name = "mobilenetv2_base"
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(settings.image_height, settings.image_width, 3))
    x = _augmentation_layers(tf, settings)(inputs)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenetv2_preprocess")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, metadata


def _prepare_mobilenet_fine_tuning(tf, model, settings: TrainingSettings) -> bool:
    try:
        base_model = model.get_layer("mobilenetv2_base")
    except ValueError:
        return False

    base_model.trainable = True
    trainable_layers = max(settings.fine_tune_layers, 0)
    if trainable_layers == 0:
        return False

    freeze_until = max(len(base_model.layers) - trainable_layers, 0)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= freeze_until
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return any(layer.trainable for layer in base_model.layers)


def _count_classes(dataset, class_names: list[str]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for _, labels in dataset:
        for label in labels.numpy().tolist():
            counts[class_names[int(label)]] += 1
    return counts


def _compute_class_weight(class_counts: dict[str, int]) -> dict[int, float]:
    total = sum(class_counts.values())
    number_of_classes = len(class_counts)
    class_weight: dict[int, float] = {}
    for index, count in enumerate(class_counts.values()):
        if count > 0:
            class_weight[index] = total / (number_of_classes * count)
    return class_weight


def _merge_histories(histories: list[Any]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


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
