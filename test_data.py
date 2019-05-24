import os
import sys
import time

import tensorflow as tf
import cv2

from datasets import data_generator
import main

flags = tf.app.flags
FLAGS = flags.FLAGS


# def get_test_model():
#     inputs = tf.keras.Input(shape=(FLAGS.train_crop_size[0], FLAGS.train_crop_size[1], 3))
#     return tf.keras.Model(inputs=inputs, outputs=inputs)


def my_train(epoch, steps_per_epoch, batch_handler, iterator):
    try:
        while True:
            row = iterator.next()
            image = row[0]
            label = row[1]
            batch_handler(image, label)
    except StopIteration as e:
        print(e)


def handle_batch(images, labels):
    for i in range(len(images)):
        # image = images[i] * 255.0
        image = tf.math.multiply(images[i], tf.constant(255.0, tf.float64))
        image = tf.to_int64(image)
        image = image.numpy()
        image = image.astype('uint8').reshape((480, 640, 3))
        label = tf.to_int64(labels[i])
        label = label.numpy()
        # filename = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
        filename = '%f' % time.time()
        cv2.circle(image, (label[0], label[1]), 5, (0, 255, 0))
        cv2.circle(image, (label[2], label[3]), 5, (0, 255, 0))
        cv2.circle(image, (label[4], label[5]), 5, (0, 255, 0))
        cv2.imwrite('/home/yjy/dataset/palmprint_dectection/LHand/palmvein_augment/' + filename + '.jpg', image)
        # print(image.numpy())
        # print(label.numpy())
        # print('---------------------------------------------')


def main(_):
    clone_batch_size = 8
    # steps_per_epoch = int(1800 / clone_batch_size)
    # m_epoch = int((FLAGS.training_number_of_steps+1800) / 1800)
    steps_per_epoch = 600 * 4
    m_epoch = 1

    train_dataset = data_generator.Dataset(
        dataset_name=FLAGS.dataset,
        split_name=FLAGS.train_split,
        dataset_dir=FLAGS.dataset_dir,
        batch_size=clone_batch_size,
        crop_size=FLAGS.train_crop_size,
        min_scale_factor=FLAGS.min_scale_factor,
        max_scale_factor=FLAGS.max_scale_factor,
        num_readers=2,
        is_training=True,
        should_shuffle=False,
        should_repeat=False)
    train_iterator = train_dataset.get_one_shot_iterator()

    my_train(m_epoch, steps_per_epoch, batch_handler=handle_batch, iterator=train_iterator)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.app.run(main=main, argv=[sys.argv[0]])
