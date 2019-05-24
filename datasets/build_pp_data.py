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

"""Converts Palmprint data to TFRecord file format with Example protos.

Palmprint dataset is expected to have the following directory structure:

  + datasets
    - build_data.py
    - build_pp_data.py (current working directory).
    + LHand
      + tfrecord
      + palmprint_trainval
        - figCon.txt
        - IMG_0109_004_RRR_019_102.bmp
    + RHand

Image folder:
  ./LHand/palmprint_trainval

finger roots labels:
  ./LHand/palmprint_trainval/figCon.txt

list folder:
  ./LHand

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/labels/coordinates: finger roots data.
"""
import os.path
import sys

import build_data
import math
import tensorflow as tf

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string('image_folder',
                           './LHand/palmprint_trainval',
                           'Folder containing images.')

tf.compat.v1.app.flags.DEFINE_string(
    'coordinates_filename_folder',
    './LHand/palmprint_trainval',
    'Folder containing coordinates labels')

tf.compat.v1.app.flags.DEFINE_string(
    'list_folder',
    './LHand',
    'Folder containing lists for training and validation')

tf.compat.v1.app.flags.DEFINE_string(
    'output_dir',
    './LHand/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

_NUM_SHARDS = 4


def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format.

    Args:
      dataset_split: The dataset split (e.g., train, test).

    """
    dataset = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset)
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    coordinate_filenames = 'figCon'
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('jpg', channels=1)  # 3

    # Read the finger roots data.
    coordinates_filename = os.path.join(
        FLAGS.coordinates_filename_folder,
        coordinate_filenames + '.' + FLAGS.label_format)
    coordinates_data = tf.compat.v1.gfile.FastGFile(coordinates_filename, 'r').read()
    label_reader = build_data.TxtReader(coordinates_data)

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, len(filenames), shard_id))
                sys.stdout.flush()

                # Read the image.
                image_filename = os.path.join(
                    FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
                image_data = tf.compat.v1.gfile.FastGFile(image_filename, 'rb').read()
                # tf.cast(image_data, tf.float64)
                # image_data = tf.math.l2_normalize(image_data)
                height, width = image_reader.read_image_dims(image_data)
                finger_roots = label_reader.read_coordinates_data(filenames[i] + '.' + FLAGS.image_format)
                print(finger_roots)
                if len(finger_roots) != 6:
                    raise ValueError("finger roots data: %s is illegal", finger_roots)

                # Convert to tf example.
                example = build_data.image_coordinates_to_tfexample(
                    image_data, filenames[i], height, width, finger_roots)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
    dataset_splits = tf.io.gfile.glob(os.path.join(FLAGS.list_folder, '*.txt'))
    for dataset_split in dataset_splits:
        _convert_dataset(dataset_split)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.app.run()
