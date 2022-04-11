# https://www.tensorflow.org/tutorials/images/classification

# To save and load Models
# https://www.tensorflow.org/tutorials/keras/save_and_load

import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

DATASET_PATH_TAR = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'


def start_sample():
    # Getting data and understanding data
    dataset_url = DATASET_PATH_TAR
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    # Loading data and prepare for train and validation
    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Build classification
    class_names = train_ds.class_names
    print(class_names)
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    # Configure data for performance
    tune = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tune)
    val_ds = val_ds.cache().prefetch(buffer_size=tune)

    # Data normalization
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    # Create model
    num_classes = len(class_names)
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Summary
    model.summary()

    # Train model
    epochs = 10
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # PREDICT
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)],
                                                                                        100 * np.max(score))
    )
