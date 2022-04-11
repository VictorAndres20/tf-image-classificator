import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os

from src.computer_vision.model_classification import ModelClassification


class ModelCV:

    def __init__(self, model_classification: ModelClassification):
        self.epochs = 10
        self.model_classification = model_classification
        self.model = None

    def compile(self):
        self.model = Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(self.model_classification.img_height,
                                                                               self.model_classification.img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.model_classification.class_names))
        ])
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def train_model(self):
        self.model.fit(
            self.model_classification.train_ds,
            validation_data=self.model_classification.val_ds,
            epochs=self.epochs
        )

    def predict(self, image_path: str, image_name: str):
        img = keras.preprocessing.image.load_img(
            os.path.join(image_path, image_name), target_size=(self.model_classification.img_height,
                                                               self.model_classification.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return {"class": self.model_classification.class_names[np.argmax(score)],
                "percent": np.max(score)}

    def save(self, file_name: str):
        self.model.save(file_name)

    def load(self, file_name: str):
        self.model = tf.keras.models.load_model(file_name)
