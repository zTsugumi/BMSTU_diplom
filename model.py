import numpy as np
import tensorflow as tf
import json


class Model(object):
    '''
    This contructs the abstract class for all type of models
    '''

    def __init__(self, name, mode, conf_path='conf.json'):
        self.name = name
        self.mode = mode
        self.conf_path = conf_path
        self.model = None
        self.load_conf()

    def load_conf(self):
        with open(self.conf_path) as f_conf:
            self.conf = json.load(f_conf)

    def load_weight(self):
        try:
            self.model.load_weigths(self.dir_model)
        except Exception as e:
            print('[WARNING] dir_model not found')

    def save_weight(self):
        try:
            self.model.save_weights(self.dir_model)
        except Exception as e:
            print('[WARNING] dir_model not found')

    def predict(self, data_set):
        return self.model.predict(data_test)

    def evaluate(self, x_test, y_test):
        y_pred, x_reconstruct = self.model.predict(x_test)
        correct = tf.reduce_sum(
            tf.argmax(y_pred, axis=1) == tf.argmax(y_test, axis=1))
        accuracy = correct / y_test.shape[0]
        test_err = 1.0 - accuracy
        print(f'Test Accuracy: {accuracy:.4%}')
        print(f'Test Error: {test_err:.4%}')
