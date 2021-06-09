from tensorflow import keras


class SimpleEpochProgressMonitor(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        print('Epoch:', end='')

    def on_train_end(self, logs=None):
        print('')

    def on_epoch_begin(self, epoch, logs=None):
        print(f' {epoch + 1}', end='')
