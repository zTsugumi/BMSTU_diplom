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


class DigitCaps(tf.keras.layers.Layer):
    '''
    This constructs a digit capsule layer
    '''

    def __init__(self, C, L, r, **kwargs):
        super().__init__(**kwargs)
        self.C = C      # C: number of digit capsules
        self.L = L      # L: digit capsules dimension (num of properties)
        self.r = r      # r: number of routing

    def build(self, input_shape):
        H = input_shape[1]
        W = input_shape[2]
        input_C = input_shape[3]
        input_L = input_shape[4]

        self.W = self.add_weight(
            shape=(H*W*input_C, input_L, self.C*self.L),
            initializer='glorot_uniform',
            name='W'
        )
        self.bias = self.add_weight(
            shape=(self.C, self.L),
            initializer='zeros',
            name='bias'
        )
        self.built = True

    def call(self, input):
        H, W, input_C, input_L = input.shape[1:]
        u = tf.reshape(input, shape=(-1, H*W*input_C, input_L))
        # Here we multiply (1,8) x (8,160)
        u_hat = tf.einsum(
            '...ji,jik->...jk', u, self.W)          # (None, H*W*input_C, C*L)
        u_hat = tf.reshape(u, shape=(
            -1, H*W*input_C, self.C, self.L))       # (None, H*W*input_C, C, L)

        # Routing
        b = tf.zeros(
            tf.shape(u_hat)[:-1])[..., None]        # (None, H*W*input_C, C, 1)
        for r in range(self.r):
            c = tf.nn.softmax(b, axis=2)            # (None, H*W*input_C, C, 1)
            s = tf.reduce_sum(
                u_hat*c, axis=1, keepdims=True)     # (None, 1, C, L)
            s += self.bias
            v = squash(s)                           # (None, 1, C, L)
            if r < self.r - 1:
                agreement = tf.reduce_sum(
                    u_hat * v, axis=-1, keepdims=True)
                b += agreement
        v = tf.squeeze(v, axis=1)                   # (None, C, L)
        return v
