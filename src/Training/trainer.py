"""
This module provides model training capabilities
"""

import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.config.experimental import (
    list_physical_devices,
    set_virtual_device_configuration,
    VirtualDeviceConfiguration,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard


class Trainer:
    """
    This class defines training methods the models train here do the
    following
    - Image classification
    - Calculating predicted probabilities
    """

    def __init__(self, labels: list, images: list):
        self.encoder = LabelEncoder()
        self.y_encoded = None
        self.y_onehot = None
        self.labels = labels
        self.images = images
        self.logs_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def encode_categorical(self):
        """
        Encodes the labels using one hot encoding
        """
        self.y_encoded = self.encoder.fit_transform(self.labels)
        self.y_onehot = to_categorical(self.y_encoded)

    def split(self):
        """
        Splits the dataset into training and test components
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.images, self.y_onehot, test_size=0.2, random_state=42
        )
        return (x_train, x_test, y_train, y_test)

    def build_cnn_model(self):
        """
        Builds the cnn model to use
        """
        gpus = list_physical_devices("GPU")
        datagen = ImageDataGenerator(
            width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True
        )

        early_stop = EarlyStopping(
            monitor="val_loss",  # Monitor validation loss
            patience=10,  # Wait for 5 epochs before stopping if no improvement
            restore_best_weights=True,
        )
        learning_rate_red = ReduceLROnPlateau(
            monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1
        )
        tensor_board = TensorBoard(log_dir=self.logs_dir, histogram_freq=1)
        # Encode the labels before training
        if gpus:
            try:
                set_virtual_device_configuration(
                    gpus[0], [VirtualDeviceConfiguration(memory_limit=4096)]
                )

                self.encode_categorical()
                model = Sequential(
                    [
                        # Build feature map and activation function and return an activation map.
                        Conv2D(
                            32, (3, 3), activation="relu", input_shape=(150, 150, 3)
                        ),
                        MaxPooling2D((2, 2)),
                        Conv2D(32, (3, 3), activation="relu"),
                        MaxPooling2D((2, 2)),
                        Conv2D(32, (3, 3), activation="relu"),
                        Flatten(),
                        Dense(64, activation="relu"),
                        Dense(64, activation="relu"),
                        Dense(8, activation="softmax"),
                    ]
                )

                # Call the splitter and obtain the x_train and y_train values
                x_train, x_test, y_train, y_test = self.split()
                datagen.fit(x_train)

                model.compile(
                    optimizer="adam",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"],
                )
                model.fit(
                    datagen.flow(x_train, y_train, batch_size=128),
                    epochs=180,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop, learning_rate_red, tensor_board],
                )
                predictions = model.predict(self.images)
                return predictions
            except RuntimeError as e:
                print(e)
