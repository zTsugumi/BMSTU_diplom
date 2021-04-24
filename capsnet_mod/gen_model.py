import tensorflow as tf
from capsnet_mod.layers import PrimaryCaps


def encoder_graph(input_shape):
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

    primary_caps = PrimaryCaps(16, 8, 9, 2)(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[primary_caps],
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

    encoder = encoder_graph(input_shape)
    primary_caps = encoder(inputs)

    encoder.summary()