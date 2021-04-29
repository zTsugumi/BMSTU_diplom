import numpy as np
import tensorflow as tf
from capsnet_mod.layers import PrimaryCaps, DigitCaps, Length, Mask

params = {
    'conv_filters': 256,
    'conv_kernel': 9,
    'conv_stride': 1,

    'caps_primary': 16,
    'caps_primary_dim': 8,
    'caps_primary_kernel': 9,
    'caps_primary_stride': 1,

    'caps_digit_dim': 16
}


def encoder_graph(input_shape, output_class):
    '''
    This constructs the Encoder layers of Modified Capsule Network
    '''
    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    primary_caps = PrimaryCaps(
        C=params['caps_primary'],
        L=params['caps_primary_dim'],
        k=params['caps_primary_kernel'],
        s=params['caps_primary_stride'])(x)
    digit_caps = DigitCaps(
        C=output_class,
        L=params['caps_digit_dim'])(primary_caps)
    digit_caps_len = Length()(digit_caps)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[primary_caps, digit_caps, digit_caps_len],
        name='Encoder'
    )


def decoder_graph(input_shape, output_class):
    '''
    This constructs the Decoder layers
    '''
    inputs = tf.keras.Input(
        output_class*params['caps_digit_dim']
    )

    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
    x = tf.keras.layers.Reshape(input_shape)(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=x,
        name='Decoder'
    )


def build_graph(input_shape, output_class, mode):
    '''
    This contructs the whole architecture of Capsule Network 
    (Encoder + Decoder)
    '''
    # Encoder
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.Input(output_class)

    encoder = encoder_graph(input_shape, output_class)
    primary_caps, digit_caps, digit_caps_len = encoder(inputs)

    encoder.summary()

    # Decoder
    if mode == 'train':
        masked = Mask()([digit_caps, y_true])
    elif mode == 'test':
        masked = Mask()(digit_caps)
    elif mode == 'exp':
        noise = tf.keras.Input(
            (output_class, params['caps_digit_dim']))
        digit_caps_noise = tf.keras.layers.add([digit_caps, noise])
        masked = Mask()([digit_caps_noise, y_true])
    else:
        raise RuntimeError('mode not recognized')

    decoder = decoder_graph(input_shape, output_class)
    x_reconstruct = decoder(masked)

    decoder.summary()

    if mode == 'train':
        return tf.keras.Model(
            inputs=[inputs, y_true],
            outputs=[digit_caps_len, x_reconstruct],
            name='CapsNetMod'
        )
    elif mode == 'test':
        return tf.keras.Model(
            inputs=[inputs],
            outputs=[digit_caps_len, x_reconstruct],
            name='CapsNetMod'
        )
    elif mode == 'exp':
        return tf.keras.Model(
            inputs=[inputs, y_true, noise],
            outputs=[digit_caps_len, x_reconstruct],
            name='CapsNetMod'
        )
    else:
        raise RuntimeError('mode not recognized')
