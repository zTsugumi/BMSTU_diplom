import numpy as np
import tensorflow as tf

SMALLNORB_CLASS = 5
SAMPLES = 24300
INPUT_SHAPE = 96
SCALE_SHAPE = 64
PACTH_SHAPE = 48
MAX_DELTA = 2.0
LOWER_CONTRAST = 0.5
UPPER_CONTRAST = 1.5
N_THREAD = 8


def pre_process(dataset):
    x = np.empty((SAMPLES, INPUT_SHAPE, INPUT_SHAPE, 2))
    y = np.empty((SAMPLES,))

    for idx, data in enumerate(dataset.batch(1)):
        x[idx, :, :, 0:1] = data['image']
        x[idx, :, :, 1:2] = data['image2']
        y[idx] = data['label_category']

    # Standardize
    # x[..., 0] = (x[..., 0] - x[..., 0].mean()) / x[..., 0].std()
    # x[..., 1] = (x[..., 1] - x[..., 1].mean()) / x[..., 1].std()
    y = tf.one_hot(y, depth=SMALLNORB_CLASS)

    return x, y


def generator(x, y):
    return (x, y), (y, x)


def standardize(x, y):
    x = tf.image.per_image_standardization(x)
    x = tf.image.resize(x, [SCALE_SHAPE, SCALE_SHAPE])
    x = tf.image.random_crop(x, [PACTH_SHAPE, PACTH_SHAPE, 2])
    x = tf.image.random_brightness(x, max_delta=MAX_DELTA)
    x = tf.image.random_contrast(x, lower=LOWER_CONTRAST, upper=UPPER_CONTRAST)
    return x, y


def generate_tf_data(x_train, y_train, x_test, y_test, batch_size):
    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    data_train = data_train.map(standardize, num_parallel_calls=N_THREAD)
    data_train = data_train.map(generator, num_parallel_calls=N_THREAD)
    data_train = data_train.batch(batch_size, drop_remainder=True)
    data_train = data_train.prefetch(-1)

    bound = (SCALE_SHAPE - PACTH_SHAPE) // 2
    x_test_patch = x_test[:, bound:-bound, bound:-bound, :]
    data_test = tf.data.Dataset.from_tensor_slices((x_test_patch, y_test))
    data_test = data_test.cache()
    data_test = data_test.map(generator, num_parallel_calls=N_THREAD)
    data_test = data_test.batch(1, drop_remainder=True)
    data_test = data_test.prefetch(-1)

    return data_train, data_test
