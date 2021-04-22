import numpy as np
import tensorflow as tf
from capsnet.layers import PrimaryCaps

params = {
    'conv_filters': 256,
    'conv_kernel': 9,
    'conv_stride': 1,

    'caps_primary': 32,
    'caps_primary_dim': 8,
    'caps_primary_kernel': 9,
    'caps_primary_stride': 2,

    'caps_digit': 5,
}


def encoder_graph(input_shape, r):
    '''
    This constructs the Encoder layers of Capsule Network
    '''
    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.Conv2D(
        filters=params['conv_filters'],
        kernel_size=params['conv_kernel'],
        strides=params['conv_stride'],
        padding='valid',
        activation='relu')(inputs)
    primary_caps = PrimaryCaps(
        C=params['caps_primary'],
        L=params['caps_primary_dim'],
        k=params['caps_primary_kernel'],
        s=params['caps_primary_stride'])(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[primary_caps],
        name='Encoder'
    )


def build_graph(input_shape, mode, r):
    '''
    This contructs the whole architecture of Capsule Network 
    (Encoder + Decoder)
    '''
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.Input(shape=(10))

    encoder = encoder_graph(input_shape, r)
    primary_caps = encoder(inputs)

    encoder.summary()

    # if mode == 'train':
    #     return tf.keras.Model(
    #         inputs=[inputs, y_true],
    #         outputs=[None]
    #     )
    # else:
    #     raise RuntimeError('mode not recognized')
