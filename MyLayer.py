import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np


class MyLayer(layers.Layer):
    '''calculate 2d argmax per channels
    input shape: (b,w,h,c)
    output shape: (b,2*c)
    '''

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(dynamic=True, **kwargs)

    def build(self, input_shape):
        # shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=shape,
        #                               initializer='uniform',
        #                               trainable=True)
        # Be sure to call this at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        print(tf.executing_eagerly())
        shape = tf.shape(inputs).numpy()
        ouput = np.zeros(shape=(shape[0], 2 * shape[-1]), dtype=np.float)

        for i in range(shape[0]):
            for j in range(shape[-1]):
                flattened = tf.keras.backend.flatten(inputs[i][:][:][j])
                argmax = tf.keras.backend.argmax(flattened).numpy()
                ouput[i][2 * j] = argmax[i] % shape[1]
                ouput[i][2 * j + 1] = argmax[i] / shape[1]

        return ouput

    def compute_output_shape(self, input_shape):
        print(tf.executing_eagerly())

        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape((shape[0], 2 * shape[-1]))
