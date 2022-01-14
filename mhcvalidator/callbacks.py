from tensorflow import keras
from mhcvalidator.fdr import calculate_qs
import numpy as np


class SimpleEpochProgressMonitor(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        print('Epoch:', end='')

    def on_train_end(self, logs=None):
        print('')

    def on_epoch_begin(self, epoch, logs=None):
        print(f' {epoch + 1}', end='')


class MonitorPSMsAtFDR(keras.callbacks.Callback):

    def __init__(self, validation_data, target_fdr: float = 0.01):
        self.target_fdr = target_fdr
        self.psms_at_fdr = []
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        self.psms_at_fdr = []

    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(self.x_val).flatten()

        qs = calculate_qs(preds[preds >= 0.7], self.y_val[preds >= 0.7], higher_better=True)
        n_psms = np.sum((qs <= self.target_fdr) & (self.y_val[preds >= 0.7] == 1))

        self.psms_at_fdr.append(n_psms)

        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs["n_psms"] = n_psms

        print(f'\nepoch {epoch} PSMs: {n_psms}')