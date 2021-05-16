import tensorflow as tf
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from utils import Dataset, plot_image, plot_image_misclass
from capsnet import CapsNet
from capsnet_mod import CapsNetMod

import traceback
import sys


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)


def msg_box(type, msg_text):
    msg = QMessageBox()
    if type == 'error':
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
    if type == 'warning':
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning")
    elif type == 'info':
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Info")

    msg.setText(msg_text)
    msg.exec_()


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('main.ui', self)

        self.threadpool = QThreadPool()

        self.progBar_Test.setMaximum(1)
        self.pb_Test_load.clicked.connect(lambda: self.load_model())
        self.pb_Test_test.clicked.connect(lambda: self.test_model())
        self.data = None
        self.model = None

    def load_model(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, 'QFileDialog.getOpenFileName()', '', 'All Files (*)', options=options)

        if not filename:
            return

        dataset = str(self.cb_Test_dataset.currentText())
        model_idx = int(self.cb_Test_model.currentIndex())

        self.progBar_Test.setMaximum(0)

        def load():
            self.data = Dataset(dataset)
            if model_idx == 0:
                self.model = CapsNet(dataset, mode='test', r=3)
            elif model_idx == 1:
                self.model = CapsNetMod(dataset, mode='test')
            return self.model.load_weight(0, filename)

        def output(ok):
            if not ok:
                self.model = None
                msg_box('warning', 'Load model failed!')
            else:
                msg_box('info', 'Load model successful!')
            self.progBar_Test.setMaximum(1)

        worker = Worker(load)
        worker.signals.result.connect(output)
        self.threadpool.start(worker)

    def test_model(self):
        if not self.data or not self.model:
            msg_box('warning', 'Model not loaded!')
            return

        self.progBar_Test.setMaximum(0)

        def test():
            acc, err = self.model.evaluate(self.data.x_test, self.data.y_test)
            y_pred = self.model.predict(self.data.x_test)[0]

            return acc, err, y_pred

        def output(res):
            acc, err, y_pred = res
            self.le_Test_acc.setText(f'{acc*100:.4f}%')
            self.le_Test_err.setText(f'{err*100:.4f}%')
            n_img = int(self.le_Test_nimg.text())
            plot_image_misclass(
                self.data.x_test, self.data.y_test, y_pred, self.data.class_names, n_img)
            self.progBar_Test.setMaximum(1)

        worker = Worker(test)
        worker.signals.result.connect(output)
        self.threadpool.start(worker)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()
