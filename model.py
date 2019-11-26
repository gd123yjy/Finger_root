import tensorflow as tf
import MyLayer
import numpy as np

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


def get_model():
    inputs = tf.keras.Input(shape=(FLAGS.train_crop_size[0], FLAGS.train_crop_size[1], FLAGS.image_channel))

    conv_1 = tf.keras.layers.Conv2D(filters=2, padding='same', kernel_size=3)(inputs)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same', data_format='channels_last')(conv_1)
    dropout_1 = tf.keras.layers.Dropout(rate=0.1)(pool_1)

    conv_2 = tf.keras.layers.Conv2D(filters=4, padding='same', kernel_size=3)(dropout_1)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same', data_format='channels_last')(conv_2)
    dropout_2 = tf.keras.layers.Dropout(rate=0.2)(pool_2)

    conv_3 = tf.keras.layers.Conv2D(filters=6, padding='same', kernel_size=3)(dropout_2)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same', data_format='channels_last')(conv_3)
    dropout_3 = tf.keras.layers.Dropout(rate=0.3)(pool_3)

    conv_4 = tf.keras.layers.Conv2D(filters=8, padding='same', kernel_size=3)(dropout_3)
    pool_4 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same', data_format='channels_last')(conv_4)
    dropout_4 = tf.keras.layers.Dropout(rate=0.3)(pool_4)

    conv_5 = tf.keras.layers.Conv2D(filters=3, padding='same', kernel_size=3)(dropout_4)

    argmax_6 = MyLayer.MyLayer(name="mylayer")(conv_5)

    multied_8 = tf.multiply(argmax_6, 16)

    # multied_8 = tf.keras.layers.Multiply()([argmax_6, tf.constant(value=16, shape=argmax_6.shape, dtype=tf.float32)])

    # flatten_5 = tf.keras.layers.Flatten(input_shape=(31, 43))(dropout_4)
    #
    # dense_5 = tf.keras.layers.Dense(1000, activation=tf.nn.leaky_relu)(flatten_5)
    # dropout_5 = tf.keras.layers.Dropout(rate=0.5)(dense_5)
    #
    # dense_6 = tf.keras.layers.Dense(1000, activation=tf.nn.leaky_relu)(dropout_5)
    #
    # dense_7 = tf.keras.layers.Dense(6, activation=tf.nn.leaky_relu)(dense_6)

    return tf.keras.Model(inputs=inputs, outputs=multied_8)


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5)
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same', data_format='channels_last')
        self.dropout_1 = tf.keras.layers.Dropout(rate=0.1)

        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=2)
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same', data_format='channels_last')
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.2)

        self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=2)
        self.pool_3 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same', data_format='channels_last')
        self.dropout_3 = tf.keras.layers.Dropout(rate=0.3)

        self.conv_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=1)
        self.pool_4 = tf.keras.layers.MaxPool2D(pool_size=2, padding='same', data_format='channels_last')
        self.dropout_4 = tf.keras.layers.Dropout(rate=0.3)

        self.flatten_5 = tf.keras.layers.Flatten(input_shape=(30, 40))
        self.dense_5 = tf.keras.layers.Dense(1000, activation=tf.nn.elu)
        self.dropout_5 = tf.keras.layers.Dropout(rate=0.5)

        self.dense_6 = tf.keras.layers.Dense(1000, activation=tf.nn.elu)

        self.dense_7 = tf.keras.layers.Dense(6, activation=None)

    def call(self, inputs, training=False):
        _conv_1 = self.conv_1(inputs)
        _pool_1 = self.pool_1(_conv_1)
        _dropout_1 = self.dropout_1(_pool_1)

        _conv_2 = self.conv_2(_dropout_1)
        _pool_2 = self.pool_2(_conv_2)
        _dropout_2 = self.dropout_2(_pool_2)

        _conv_3 = self.conv_3(_dropout_2)
        _pool_3 = self.pool_3(_conv_3)
        _dropout_3 = self.dropout_3(_pool_3)

        _conv_4 = self.conv_4(_dropout_3)
        _pool_4 = self.pool_4(_conv_4)
        _dropout_4 = self.dropout_4(_pool_4)

        _flatten_5 = self.flatten_5(_dropout_4)
        _dense_5 = self.dense_5(_flatten_5)
        _dropout_5 = self.dropout_5(_dense_5)

        _dense_6 = self.dense_6(_dropout_5)

        return self.dense_7(_dense_6)
