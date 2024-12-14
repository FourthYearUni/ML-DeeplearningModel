"""
This module provides model training capabilities
"""

import datetime
import gc

from numpy import argmax, vstack, hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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
        self.aug_x_train = [] # Augmented features
        self.aug_y_train = [] # Augmented labels

    def encode_categorical(self):
        """
        Encodes the labels using one hot encoding
        """
        self.y_encoded = self.encoder.fit_transform(self.labels)
        self.y_onehot = to_categorical(self.y_encoded)

    def split(self, X, Y):
        """
        Splits the dataset into training and test components
        """
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        return (x_train, x_test, y_train, y_test)
    
    def set_augmented_features_labels(self, x_train, y_train):
        """
        Obtains the augmented features and labels
        """
        augmented_features = []
        augmented_labels = []
        datagen = ImageDataGenerator(
            width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True
        )
        generator = datagen.flow(x_train, y_train, batch_size=128)

        for _ in range(len(x_train) // 128):
            batch_x, batch_y = next(generator)
           # batch_features = model.predict(batch_x, verbos=0)
            augmented_features.append(batch_x)
            augmented_labels.append(batch_y)

        self.aug_x_train = augmented_features
        self.aug_y_train = augmented_labels

        # Deallocate memory so that we can optimise resource usage
        del augmented_labels
        del augmented_features
        # Manually trigger garbage collection
        gc.collect()

    def train_with_xgboost(self, augmented_labels, augmented_features):
        """
        Uses the obtained labels and features and trains the
        xgboost classifier model.
        """
        augmented_labels = argmax(augmented_labels, axis=1)
        xg_x_train, xg_x_val, xg_y_train, xg_y_val = self.split(augmented_features, augmented_labels)

        xgb_model = XGBClassifier(
                use_label_encoder=False,
                objective='multi:softmax',
                num_classes=8,
                eval_metric='mlogloss'
        )
        xgb_model.fit(xg_x_train, xg_y_train)

        # Model evaluation
        y_pred = xgb_model.predict(xg_x_val)
        accuracy = accuracy_score(xg_y_val, y_pred)
        return accuracy
    
    def build_cnn_model(self):
        """
        Builds the cnn model to use
        """
        gpus = list_physical_devices("GPU")
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
                        MaxPooling2D((2, 2)),
                        Conv2D(32, (3, 3), activation="relu"),
                        Flatten(),
                    ]
                )

                #datagen = ImageDataGenerator(
                #    width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True
                #)
                print(self.aug_x_train)
                model.compile(
                    optimizer="adam",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"],
                )
                #model.fit(
                #    self.aug_x_train,
                #    self.aug_y_train,
                #    epochs=180,
                #    validation_data=(x_test, y_test),
                #    callbacks=[early_stop, learning_rate_red, tensor_board],
                # )
                self.aug_x_train = model.predict(self.aug_x_train)
            except RuntimeError as e:
                print(e)
