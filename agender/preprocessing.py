from __future__ import annotations

import tensorflow as tf


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def decode_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, (image_size, image_size))
    return tf.cast(image, tf.float32)
