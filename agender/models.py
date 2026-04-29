from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers

from .metrics import BinaryF1Score


@dataclass(frozen=True)
class BackboneFactory:
    name: str
    builder: type[tf.keras.Model]
    preprocess_input: Callable


AVAILABLE_BACKBONES: dict[str, BackboneFactory] = {
    "efficientnetb0": BackboneFactory(
        name="EfficientNetB0",
        builder=tf.keras.applications.EfficientNetB0,
        preprocess_input=tf.keras.applications.efficientnet.preprocess_input,
    ),
    "resnet50": BackboneFactory(
        name="ResNet50",
        builder=tf.keras.applications.ResNet50,
        preprocess_input=tf.keras.applications.resnet50.preprocess_input,
    ),
    "mobilenetv3large": BackboneFactory(
        name="MobileNetV3Large",
        builder=tf.keras.applications.MobileNetV3Large,
        preprocess_input=tf.keras.applications.mobilenet_v3.preprocess_input,
    ),
}


def build_multitask_model(
    backbone_name: str,
    image_size: int,
    weights: str = "imagenet",
    pooling: str = "avg",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    key = backbone_name.lower()
    if key not in AVAILABLE_BACKBONES:
        supported = ", ".join(sorted(AVAILABLE_BACKBONES))
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Available: {supported}")

    backbone_factory = AVAILABLE_BACKBONES[key]
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")

    x = backbone_factory.preprocess_input(inputs)
    backbone = backbone_factory.builder(
        include_top=False,
        weights=weights,
        pooling=pooling,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    shared = layers.BatchNormalization(name="shared_bn")(x)
    shared = layers.Dense(256, activation="relu", name="shared_dense")(shared)
    shared = layers.Dropout(0.3, name="shared_dropout")(shared)

    gender = layers.Dense(128, activation="relu", name="gender_dense")(shared)
    gender = layers.Dropout(0.2, name="gender_dropout")(gender)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender_output")(gender)

    age = layers.Dense(128, activation="relu", name="age_dense")(shared)
    age = layers.Dropout(0.2, name="age_dropout")(age)
    age_output = layers.Dense(1, activation="linear", name="age_output")(age)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={"gender_output": gender_output, "age_output": age_output},
        name=f"{backbone_factory.name}_AgeGender",
    )
    return model, backbone


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    gender_loss_weight: float,
    age_loss_weight: float,
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "gender_output": tf.keras.losses.BinaryCrossentropy(),
            "age_output": tf.keras.losses.Huber(),
        },
        loss_weights={
            "gender_output": gender_loss_weight,
            "age_output": age_loss_weight,
        },
        metrics={
            "gender_output": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                BinaryF1Score(name="f1"),
            ],
            "age_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )


def configure_fine_tuning(
    backbone: tf.keras.Model,
    fine_tune_layers: int,
    freeze_batch_norm: bool,
):
    backbone.trainable = True
    if fine_tune_layers <= 0:
        trainable_layers = backbone.layers
    else:
        trainable_layers = backbone.layers[-fine_tune_layers:]
        for layer in backbone.layers[:-fine_tune_layers]:
            layer.trainable = False

    for layer in trainable_layers:
        if freeze_batch_norm and isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
