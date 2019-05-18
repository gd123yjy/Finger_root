"""Prepares the data used for Palmprint training/evaluation."""
import tensorflow as tf

from core import preprocess_utils


def preprocess_image_and_label_yjy(image, label, scale_factor=1.0, is_training=True):
    if is_training and label is None:
        raise ValueError('During training, label must be provided.')

    # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)

    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, scale_factor)
    processed_image.set_shape([None, None, 3])

    if label is not None:
        label.set_shape([6])

    return original_image, processed_image, label
