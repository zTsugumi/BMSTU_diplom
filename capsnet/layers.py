import numpy as np
import tensorflow as tf

EPS = tf.keras.backend.epsilon()


def safe_norm(s, axis=-1):
    '''
    Calculation of norm as tf.norm(), but here we add a small value of eps 
    to the result to avoid 0
    '''
    s_ = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    return tf.sqrt(s_ + EPS)


def squash(s):
    '''
    Squash activation
    '''
    norm = safe_norm(s, axis=-1)
    norm_squared = tf.square(norm)
    return norm_squared / (1.0 + norm_squared) / norm * s


class PrimaryCaps(tf.keras.layers.Layer):
    '''
    This constructs a primary capsule layer using regular convolution layer
    Input:
        (None, 20, 20, 256)
    Output:
        (None, 6, 6, 32, 8)
    '''

    def __init__(self, C, L, k, s, **kwargs):
        super().__init__(**kwargs)
        self.C = C      # C: number of primary capsules
        self.L = L      # L: primary capsules dimension (num of properties)
        self.k = k      # k: kernel dimension
        self.s = s      # s: stride

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.k, self.k, input_shape[-1], self.C*self.L),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.C, self.L),
            initializer='zeros',
            name='bias'
        )
        self.built = True

    def call(self, input):
        x = tf.nn.conv2d(
            input,
            filters=self.kernel,
            strides=self.s,
            padding='VALID')
        H, W = x.shape[1:3]
        x = tf.keras.layers.Reshape((H, W, self.C, self.L))(x)
        x /= self.C     # ??
        x += self.bias  # ??
        x = squash(x)
        return x
