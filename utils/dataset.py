import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import json
from utils import preprocess_mnist, preprocess_smallnorb


class Dataset(object):
    '''
    This constructs functions to process dataset
    '''

    def __init__(self, data_name, conf_path='conf.json'):
        self.data_name = data_name
        self.conf_path = conf_path
        self.load_config()
        self.load_dataset()

    def load_config(self):
        with open(self.conf_path) as f_conf:
            self.conf = json.load(f_conf)

    def load_dataset(self):
        if self.data_name == 'MNIST':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = \
                tf.keras.datasets.mnist.load_data()
            self.x_train, self.y_train = preprocess_mnist.pre_process(
                self.x_train, self.y_train)
            self.x_test, self.y_test = preprocess_mnist.pre_process(
                self.x_test, self.y_test)
            self.class_names = list(range(10))
        elif self.data_name == 'SMALLNORB':
            (data_train, data_test), data_info = tfds.load(
                name='smallnorb',
                split=['train', 'test'],
                data_dir=self.conf['dir_data'],
                shuffle_files=True,
                as_supervised=False,
                with_info=True
            )
            self.x_train, self.y_train = preprocess_smallnorb.pre_process(
                data_train)
            self.x_test, self.y_test = preprocess_smallnorb.pre_process(
                data_test)
            self.x_test, self.y_test = preprocess_smallnorb.pre_process_test(
                self.x_test, self.y_test)
            self.class_names = data_info.features['label_category'].names
        else:
            raise RuntimeError('data_name not recognized')

    def get_tf_data(self):
        if self.data_name == 'MNIST':
            data_train, data_test = preprocess_mnist.generate_tf_data(
                self.x_train, self.y_train,
                self.x_test, self.y_test,
                self.conf['batch_size']
            )
        elif self.data_name == 'SMALLNORB':
            data_train, data_test = preprocess_smallnorb.generate_tf_data(
                self.x_train, self.y_train,
                self.x_test, self.y_test,
                self.conf['batch_size']
            )
        else:
            raise RuntimeError('data_name not recognized')

        return data_train, data_test
