"""
This module provides model training capabilities
"""

import datetime

from keras.src.models.model import Model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.config.experimental import (
    list_physical_devices,
    set_virtual_device_configuration,
    VirtualDeviceConfiguration,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from scikeras.wrappers import KerasClassifier
from numpy import argmax, bincount


def build_cnn_model(
    optimizer="adam", batch_size=32, epocs=20, learning_rate=1e-05
) -> Model:
    """
    Builds the cnn model to use
    """

    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    model = Sequential(
        [
            # Build feature map and activation function and return an activation map.
            Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(8, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


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
            self.images,
            self.y_onehot,
            test_size=0.2,
            random_state=42,
            stratify=self.y_onehot,
        )
        x_val, x_Test, y_val, y_Test = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        print(argmax(y_train), y_train)
        print("Train set bincount", bincount(argmax(y_train, axis=1)))
        print("Test set bincount", bincount(argmax(y_Test, axis=1)))
        print("Validation set bincount", bincount(argmax(y_val, axis=1)))
        return (x_train, x_Test, y_train, y_Test, x_val, y_val)

    def train(self, x_train, y_train, x_test, y_test):
        """
        This functions runs and compiles a prebuilt and precompiled model.
        """
        gpus = list_physical_devices("GPU")

        if gpus:
            set_virtual_device_configuration(
                gpus[0], [VirtualDeviceConfiguration(memory_limit=4096)]
            )

        hyper_parameter_grid = {
            "optimizer": ["adam", "sgd"],
            "batch_size": [16, 32, 48],
            "epocs": [50, 100, 150],
            "learning_rate": [0.00001, 0.0001, 0.001, 0.01],
        }
        model = KerasClassifier(
            model=build_cnn_model,
            epocs=20,
            batch_size=32,
            verbose=1,
            learning_rate=1e-05,
        )
        grid = GridSearchCV(
            estimator=model,
            param_grid=hyper_parameter_grid,
            error_score="raise",
            scoring="accuracy",
            cv=3,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",  # Monitor validation loss
            patience=10,  # Wait for 5 epochs before stopping if no improvement
            restore_best_weights=True,
        )
        learning_rate_red = ReduceLROnPlateau(
            monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6, verbose=3
        )
        tensor_board = TensorBoard(log_dir=self.logs_dir, histogram_freq=1)
        datagen = ImageDataGenerator(
            width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True
        )
        datagen.fit(x_train)
        aug_x_train, aug_y_train = next(
            datagen.flow(x_train, y_train, batch_size=len(x_train))
        )
        grid.fit(
            aug_x_train,
            aug_y_train,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, learning_rate_red, tensor_board],
        )
        best_model = grid.best_estimator_
        print(grid.best_params_)
        # predictions = best_model.predict(x_train, batch_size=32)
        # return predictions
