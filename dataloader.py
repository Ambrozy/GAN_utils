import os
import numpy as np
import tensorflow.keras.utils as U
import tensorflow.keras.preprocessing as P


class FolderImageGenerator(U.Sequence):
    def __init__(self, x_folder, y_folder, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.x_files = [x_folder + name for name in sorted(os.listdir(x_folder))]
        self.y_files = [y_folder + name for name in sorted(os.listdir(y_folder))]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.x_files) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = [np.array(P.image.load_img(self.x_files[i])) / 255.0 for i in indexes]
        Y = [np.array(P.image.load_img(self.y_files[i])) / 127.5 - 1.0 for i in indexes]

        X = np.array(X)  # [0...1]
        Y = np.array(Y)  # [-1...1]
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)
