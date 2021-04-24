import tensorflow as tf

EPS = tf.keras.backend.epsilon()


def safe_norm(s, axis=-1, keepdims=True):
    '''
    Calculation of norm as tf.norm(), but here we add a small value of eps 
    to the result to avoid 0
    '''
    s_ = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
    return tf.sqrt(s_ + EPS)


def squash(s):
    '''
    Squash activation
    '''
    norm = safe_norm(s, axis=-1)
    return (1.0 - 1.0/tf.exp(norm)) * (s / norm)


class PrimaryCaps(tf.keras.layers.Layer):
    '''
    This constructs a primary capsule layer using regular convolution layer
    '''

    def __init__(self, C, L, k, s, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.C = C      # C: number of primary capsules
        self.L = L      # L: primary capsules dimension (num of properties)
        self.k = k      # k: kernel dimension
        self.s = s      # s: stride

    def get_config(self):
        config = {
            'C': self.C,
            'L': self.L,
            'k': self.k,
            's': self.s,
        }
        base_config = super(PrimaryCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def build(self, input_shape):
        self.DW_Conv = tf.keras.layers.Conv2D(
            filters=self.C*self.L,
            kernel_size=self.k,
            strides=self.s,
            kernel_initializer='glorot_uniform',
            padding='valid',
            groups=self.C*self.L,
            activation='linear',
            name='conv'
        )
        self.built = True

    def call(self, input):
        x = self.DW_Conv(input)
        x = tf.keras.layers.Reshape((self.C, self.L))(x)
        x = squash(x)
        return x
