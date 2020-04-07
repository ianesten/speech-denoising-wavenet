# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Layers.py

import keras
import tensorflow as tf


class AddSingletonDepth(keras.layers.Layer):

    def call(self, x, mask=None):
        x = keras.backend.expand_dims(x, -1)  # add a dimension of the right

        if keras.backend.ndim(x) == 4:
            return keras.backend.permute_dimensions(x, (0, 3, 1, 2))
        else:
            return x

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return input_shape[0], 1, input_shape[1], input_shape[2]
        else:
            return input_shape[0], input_shape[1], 1


class Subtract(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Subtract, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[0] - x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Slice(keras.layers.Layer):

    def __init__(self, selector, output_shape, **kwargs):
        self.selector = selector
        self.desired_output_shape = output_shape
        super(Slice, self).__init__(**kwargs)

    def call(self, x, mask=None):

        selector = self.selector

        assert len(selector) == 2

        if not type(selector[1]) is slice and not type(selector[1]) is int:
            x = tf.transpose(x, [0, 2, 1])
            y = x[(selector[1], selector[0])]
            return tf.transpose(y, [0, 2, 1])
        return x[selector]


    def compute_output_shape(self, input_shape):

        output_shape = (None,)
        for i, dim_length in enumerate(self.desired_output_shape):
            if dim_length == Ellipsis:
                output_shape = output_shape + (input_shape[i+1],)
            else:
                output_shape = output_shape + (dim_length,)
        return output_shape
