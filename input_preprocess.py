"""Prepares the data used for Palmprint training/evaluation."""
import tensorflow as tf

from core import preprocess_utils

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


def preprocess_image_and_label_yjy(image, label, crop_height, crop_width,
                                   is_rotate, is_scale,
                                   min_scale_factor=1., max_scale_factor=1.,
                                   is_training=True):
    if is_training and label is None:
        raise ValueError('During training, label must be provided.')

    # Keep reference to original image and label.
    original_image = image
    original_label = label

    processed_image = tf.cast(image, tf.float32)
    processed_label = tf.cast(label, tf.float32)

    if is_scale:
        # randomly scale
        scale_factor = preprocess_utils.get_random_scale(
            min_scale_factor, max_scale_factor, 0)
        processed_image, processed_label = preprocess_utils.randomly_scale_image_and_label(
            processed_image, processed_label, scale_factor)
        processed_image.set_shape([None, None, FLAGS.image_channel])

        # Pad image to have dimensions >= [crop_height, crop_width]
        image_shape = tf.shape(input=processed_image)
        image_height = image_shape[0]
        image_width = image_shape[1]

        target_height = image_height + tf.maximum(crop_height - image_height, 0)
        target_width = image_width + tf.maximum(crop_width - image_width, 0)

        # Pad image with mean pixel value.
        mean_pixel = tf.reshape([127.5, 127.5, 127.5], [1, 1, 3])  # [127.5, 127.5, 127.5]
        processed_image = preprocess_utils.pad_to_bounding_box(
            processed_image, 0, 0, target_height, target_width, mean_pixel)

        # crop to [crop_height,crop_width]
        processed_image, processed_label = preprocess_utils.random_crop(processed_image, processed_label, crop_height,
                                                                        crop_width)

    # todo: rotate factor should be determined by command line
    if is_rotate:
        # randomly rotate
        rotate_factor = preprocess_utils.get_rotate_scale(min_rotate_factor=0, max_rotate_factor=359, step_size=90)
        processed_image, processed_label = preprocess_utils.randomly_rotate_image_and_label(
            processed_image, processed_label, rotate_factor)
        processed_image.set_shape([None, None, 3])

    if processed_label is not None:
        processed_label.set_shape([6])
    if original_label is not None:
        original_label.set_shape([6])

    return original_image, processed_image, original_label, processed_label
