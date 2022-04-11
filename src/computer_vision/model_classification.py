import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ModelClassification:

    def __init__(self, dataset_path: str):
        self.batch_size = 32
        self.img_height = 180
        self.img_width = 180

        self.dataset_path = dataset_path
        self.train_ds = None
        self.val_ds = None
        self.class_names = None
        self.build_class_names()

    def build_train_ds(self):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

    def build_val_ds(self):
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

    def build_class_names(self):
        self.build_train_ds()
        self.build_val_ds()
        self.class_names = self.train_ds.class_names
        print(self.class_names)
        for image_batch, labels_batch in self.train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

    def normalize_data(self):
        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
        normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixels values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))
