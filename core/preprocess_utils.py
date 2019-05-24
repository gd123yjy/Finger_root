# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions related to preprocessing inputs."""
import tensorflow as tf
import math


def _crop_label(label, offset_height, offset_width):
    """
    label=[w,h,w,h,w,h]
    :param label:
    :param offset_height:
    :param offset_width:
    :return:
    """
    return label - [offset_width, offset_height, offset_width, offset_height, offset_width, offset_height]


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      The cropped (and resized) image.

    Raises:
      ValueError: if `image` doesn't have rank of 3.
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(input=image)

    if len(image.get_shape().as_list()) != 3:
        raise ValueError('input must have rank of 3')
    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), dtype=tf.int32)

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image


def random_crop(image, label, crop_height, crop_width):
    """Crops the given image and label.

      The function applies crop to image and label.

      Args:
        image:
        label:
        crop_height: the new height.
        crop_width: the new width.

      Returns:
        the cropped image and label.

      Raises:
        ValueError: if the images are smaller than the crop dimensions.
      """
    if image is None:
        raise ValueError('Empty image.')

    # Compute the rank assertions.
    image_rank = tf.rank(image)
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image.name, 3, image_rank])

    with tf.control_dependencies([rank_assert]):
        image_shape = tf.shape(input=image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assert, crop_size_assert]

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    # todo: offset should be 0,because it has to be symmetric with the case that scaling smaller
    # offset_height = tf.random_uniform(
    #     [], maxval=max_offset_height, dtype=tf.int32)
    # offset_width = tf.random_uniform(
    #     [], maxval=max_offset_width, dtype=tf.int32)
    offset_height = 0
    offset_width = 0

    return _crop(image, offset_height, offset_width, crop_height, crop_width), _crop_label(label, offset_height,
                                                                                           offset_width)


# todo: return value should not be only 0 or 2
def get_rotate_scale(min_rotate_factor, max_rotate_factor, step_size):
    """Gets a random rotate value.

      Args:
        min_rotate_factor: Minimum rotate value.
        max_rotate_factor: Maximum rotate value.
        step_size: The step size from minimum to maximum value.

      Returns:
        A random rotate value selected between minimum and maximum value.(should be)
        yet now we can only return emun{0,2} because it is hard to calculate label when rotate_factor = 1

      Raises:
        ValueError: min_rotate_factor has unexpected value.
      """
    if min_rotate_factor < 0 or min_rotate_factor > max_rotate_factor:
        raise ValueError('Unexpected value of min_rotate_factor.')

    if min_rotate_factor == max_rotate_factor:
        return tf.cast(min_rotate_factor, dtype=tf.float32)

    if step_size == 0:
        return tf.random.uniform([1],
                                 minval=min_rotate_factor,
                                 maxval=max_rotate_factor)
    # num_steps = int((max_rotate_factor - min_rotate_factor) / step_size + 1)
    # rotate_factors = tf.lin_space(min_rotate_factor / float(step_size), max_rotate_factor / float(step_size), num_steps)
    # shuffled_rotate_factors = tf.random_shuffle(rotate_factors)
    rotate_factors = [0, 2]
    shuffled_rotate_factors = tf.random.shuffle(rotate_factors)
    return tf.cast(shuffled_rotate_factors[0], dtype=tf.int32)


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.

      Args:
        min_scale_factor: Minimum scale value.
        max_scale_factor: Maximum scale value.
        step_size: The step size from minimum to maximum value.

      Returns:
        A random scale value selected between minimum and maximum value.

      Raises:
        ValueError: min_scale_factor has unexpected value.
      """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.cast(min_scale_factor, dtype=tf.float32)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random.uniform([1],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.linspace(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random.shuffle(scale_factors)
    return shuffled_scale_factors[0]


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """Pads the given image with the given pad_value.

    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes are not
    known during graph construction.

    Args:
      image: 3-D tensor with shape [height, width, channels]
      offset_height: Number of rows of zeros to add on top.
      offset_width: Number of columns of zeros to add on the left.
      target_height: Height of output image.
      target_width: Width of output image.
      pad_value: Value to pad the image tensor with.

    Returns:
      3-D tensor of shape [target_height, target_width, channels].

    Raises:
      ValueError: If the shape of image is incompatible with the offset_* or
      target_* arguments.
    """
    image_rank = tf.rank(image)
    image_rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong image tensor rank [Expected] [Actual]',
         3, image_rank])
    with tf.control_dependencies([image_rank_assert]):
        image -= pad_value
    image_shape = tf.shape(input=image)
    height, width = image_shape[0], image_shape[1]
    target_width_assert = tf.Assert(
        tf.greater_equal(
            target_width, width),
        ['target_width must be >= width'])
    target_height_assert = tf.Assert(
        tf.greater_equal(target_height, height),
        ['target_height must be >= height'])
    with tf.control_dependencies([target_width_assert]):
        after_padding_width = target_width - offset_width - width
    with tf.control_dependencies([target_height_assert]):
        after_padding_height = target_height - offset_height - height
    offset_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(after_padding_width, 0),
            tf.greater_equal(after_padding_height, 0)),
        ['target size not possible with the given target offsets'])

    height_params = tf.stack([offset_height, after_padding_height])
    width_params = tf.stack([offset_width, after_padding_width])
    channel_params = tf.stack([0, 0])
    with tf.control_dependencies([offset_assert]):
        paddings = tf.stack([height_params, width_params, channel_params])
    padded = tf.pad(tensor=image, paddings=paddings)
    return padded + pad_value


# todo: counter-clockwise rotate,rotate_factor=1 is hard to achieve
def randomly_rotate_image_and_label(image, label, rotate_factor=0):
    """counter-clockwise rotate image and label by 90*rotate_factor degree.

    Args:
      image: Image with shape [height, width, 3].
      label: Label with shape [6]: [w,h,w,h,w,h]
      rotate_factor: a tensor whose dtype=tf.int32

    Returns:
      rotated image and label.
    """
    image_shape = tf.shape(input=image)
    middle = [image_shape[1] / 2, image_shape[0] / 2,
              image_shape[1] / 2, image_shape[0] / 2,
              image_shape[1] / 2, image_shape[0] / 2]
    middle = tf.cast(middle, dtype=tf.float32)
    middle = tf.reshape(middle, (6, 1))
    label = tf.cast(label, dtype=tf.float32)
    label = tf.reshape(label, (6, 1))
    cos = tf.math.cos(tf.cast(rotate_factor, dtype=tf.float32) * math.pi / 2)
    sin = tf.math.sin(tf.cast(rotate_factor, dtype=tf.float32) * math.pi / 2)
    rotate_matrix = tf.convert_to_tensor(value=[[cos, sin, 0, 0, 0, 0],
                                          [-sin, cos, 0, 0, 0, 0],
                                          [0, 0, cos, sin, 0, 0],
                                          [0, 0, -sin, cos, 0, 0],
                                          [0, 0, 0, 0, cos, sin],
                                          [0, 0, 0, 0, -sin, cos]],
                                         dtype=tf.float32)

    image = tf.image.rot90(image, rotate_factor)
    label = tf.matmul(rotate_matrix, (label - middle)) + middle
    return image, tf.reshape(label, (6,))


def randomly_scale_image_and_label(image, label=None, scale=1.0):
    """Randomly scales image and label.

    Args:
      image: Image with shape [height, width, 3].
      label: Label with shape [6].
      scale: The value to scale image and label.

    Returns:
      Scaled image and label.
    """
    # No random scaling if scale == 1.
    if scale == 1.0:
        return image, label
    image_shape = tf.shape(input=image)
    new_dim = tf.cast(tf.cast([image_shape[0], image_shape[1]], dtype=tf.float32) * scale, dtype=tf.int32)

    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = tf.squeeze(tf.image.resize(tf.expand_dims(image, 0),
                                       new_dim, method=tf.image.ResizeMethod.BILINEAR), [0])

    label = tf.cast(label, tf.float32)
    label = tf.multiply(label, scale)
    # label = tf.cast(label, tf.int32)

    return image, label


def resolve_shape(tensor, rank=None, scope=None):
    """Fully resolves the shape of a Tensor.

    Use as much as possible the shape components already known during graph
    creation and resolve the remaining ones during runtime.

    Args:
      tensor: Input tensor whose shape we query.
      rank: The rank of the tensor, provided that we know it.
      scope: Optional name scope.

    Returns:
      shape: The full shape of the tensor.
    """
    with tf.compat.v1.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()

        if None in shape:
            shape_dynamic = tf.shape(input=tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

        return shape
