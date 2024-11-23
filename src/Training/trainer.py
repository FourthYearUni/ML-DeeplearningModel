"""
This module provides model training capabilities
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


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
        model = Sequential(
            # Build feature map and activation function and return an activation map.
            Conv2D(32, (3, 3), activaton='relu', input_shape=(150, 150, 3)),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(64, activateion='relu'),
            Dense(8, activation='softmax')
        )

        # Call the splitter and obtain the x_train and y_train values
        x_train, x_test, y_train, y_test = split()

        model.compile(optimizer='adam', loss='categorical_cross_entropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

        return model.predict(self.images)
