import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np


@tf.custom_gradient
def custom_op(x):
    shape = tf.shape(x).numpy()
    output = np.zeros(shape=(shape[0], 2 * shape[-1]), dtype=np.float)

    for i in range(shape[0]):
        for j in range(shape[-1]):
            flattened = tf.keras.backend.flatten(x[i][:][:][j])
            argmax = tf.keras.backend.argmax(flattened).numpy()
            output[i][2 * j] = argmax % shape[1]
            output[i][2 * j + 1] = argmax / shape[1]

    output = output.astype(np.int)

    def custom_grad(dy):
        def region_grad(dy):
            result = np.zeros_like(x.numpy())
            dy = dy.numpy()
            dy = np.sqrt(np.abs(dy))
            for i in range(shape[0]):
                for l in range(shape[3]):
                    low_bound = (int)(output[i][2 * l] - dy[i][2 * l] / 2)
                    up_bound = (int)(output[i][2 * l] + dy[i][2 * l] / 2)
                    if low_bound < 0: low_bound = 0
                    if up_bound > shape[1]: up_bound = shape[1]
                    for j in range(low_bound, up_bound):
                        low_bound = (int)(output[i][2 * l + 1] - dy[i][2 * l + 1] / 2)
                        up_bound = (int)(output[i][2 * l + 1] + dy[i][2 * l + 1] / 2)
                        if low_bound < 0: low_bound = 0
                        if up_bound > shape[2]: up_bound = shape[2]
                        for k in range(low_bound, up_bound):
                            result[i][j][k][l] = 1
            result = tf.convert_to_tensor(result)
            return result

        def mean_grad(dy):
            dy = dy.numpy()
            sum = 0
            for i in range(np.shape(dy)[1]):
                sum += dy[0][i]
            result = np.full_like(x.numpy(), sum)
            result = tf.convert_to_tensor(result / (shape[1] * shape[2] * 6))
            return result

        def argmax_grad(dy):
            dy = dy.numpy()
            result = np.zeros_like(x.numpy())
            for b in range(np.shape(dy)[0]):
                for i in range((int)(np.shape(dy)[1] / 2)):
                    x_i = output[b][2 * i]
                    y_i = output[b][2 * i + 1]
                    result[b][x_i][y_i][i] = (dy[b][2 * i] + dy[b][2 * i + 1]) / 2
            result = tf.convert_to_tensor(result)
            return result

        return argmax_grad(dy)

    return tf.convert_to_tensor(output, dtype=tf.float32), custom_grad


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
        return custom_op(inputs)

    def compute_output_shape(self, input_shape):
        print(tf.executing_eagerly())
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape((shape[0], 2 * shape[-1]))
