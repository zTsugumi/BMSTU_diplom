import numpy as np
import tensorflow as tf
# import os
from model import Model
from utils import checkpoint, lr_sched
from capsnet.gen_model import build_graph
from capsnet.loss import margin_loss


class CapsNet(Model):
    def __init__(self, name, mode, conf_path='conf.json', r=3):
        Model.__init__(self, name, mode, conf_path)
        self.load_conf()

        self.dir_model = self.conf['dir_log'] + f'/capsnet_{self.name}'
        self.dir_log = self.conf['dir_log'] + f'/capsnet_{self.name}'

        if name == 'MNIST':
            self.model = build_graph(
                self.conf['input_mnist'], self.conf['class_mnist'], self.mode, r)
        elif name == 'SMALLNORB':
            self.model = build_graph(
                self.conf['input_smallnorb'], self.conf['class_smallnorb'], self.mode, r)
        else:
            raise RuntimeError('name not recognized')

    def train(self, dataset, initial_epoch=0):
        data_train, data_test = dataset.get_tf_data()

        cp, tb = checkpoint(self.dir_model, self.dir_log)
        lr = lr_sched(self.conf['lr'], self.conf['lr_decay'])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.conf['lr']),
            loss=[margin_loss, 'mse'],
            loss_weights=[1.0, self.conf['alpha']],
            metrics={'Encoder': 'accuracy'}
        )

        if initial_epoch > 0:
            self.load_weight(initial_epoch)

        history = self.model.fit(
            data_train,
            validation_data=(data_test),
            epochs=self.conf['epochs'],
            batch_size=self.conf['batch_size'],
            initial_epoch=initial_epoch,

            callbacks=[cp, tb, lr]
        )

        return history
