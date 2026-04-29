from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="agender")
class BinaryF1Score(tf.keras.metrics.Metric):
    """Thresholded binary F1 for the gender head."""

    def __init__(self, name: str = "f1", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        numerator = 2.0 * self.true_positives
        denominator = numerator + self.false_positives + self.false_negatives
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
