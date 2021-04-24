import tensorflow as tf
from model import Model
from capsnet_mod.gen_model import build_graph


class CapsNetMod(Model):
    def __init__(self, name, mode, conf_path='conf.json'):
        Model.__init__(self, name, mode, conf_path)
        self.load_conf()

        self.dir_model = self.conf['dir_log'] + f'/capsnet_mod_{self.name}'
        self.dir_log = self.conf['dir_log'] + f'/capsnet_mod_{self.name}'

        if name == 'MNIST':
            self.model = build_graph(
                self.conf['input_mnist'], self.conf['class_mnist'], self.mode)
        elif name == 'SMALLNORB':
            self.model = build_graph(
                self.conf['input_smallnorb'], self.conf['class_smallnorb'], self.mode)
        else:
            raise RuntimeError('name not recognized')

    def train(self, dataset, initial_epoch=0):
        pass
