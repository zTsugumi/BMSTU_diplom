import numpy as np
import tensorflow as tf
import os
from model import Model
from capsnet.gen_model import build_graph


class CapsNet(Model):
    def __init__(self, name, mode, conf_path='conf.json', r=3):
        Model.__init__(self, name, mode, conf_path)
        self.r = r
        self.load_conf()
        self.dir_model = os.path.join(
            self.conf['dir_model'], f"capsnet_{self.name}.h5")
        self.dir_log = os.path.join(
            self.conf['dir_log'], f"capsnet_{self.name}")
        self.model = build_graph(
            self.conf['input_mnist'], self.mode, self.r)

    def train(self, dataset, init_epoch=0):
        data_train, data_test = dataset.get_tf_data()

        # self.model.compile(
        #     optimizer=tf.keras.optimizers.Adam(lr=self.conf['lr'])

        # )
