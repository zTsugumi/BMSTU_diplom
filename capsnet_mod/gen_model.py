import tensorflow as tf
from capsnet_mod.layers import PrimaryCaps, DigitCaps, Length


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

    primary_caps = PrimaryCaps(16, 8, 9, 1)(x)
    digit_caps = DigitCaps(output_class, 16)(primary_caps)
    digit_caps_len = Length()(digit_caps)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[primary_caps, digit_caps, digit_caps_len],
        name='Encoder'
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
