"""
This module handles data cleaning and labelling.
@author: Alain Mugisha (U2083264)
"""

import os
import shutil
from pathlib import Path

import cv2
from pandas import read_excel
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import array
from cleanlab import Datalab

from ..Utils.files import File
from src.Utils.labeller import Labeller
from ..Utils.resize import resize_picture


class Cleaner:
    """
    This class handles data cleaning and labelling.
    """

    def __init__(self) -> None:
        self.data_folder = Path(__file__).parent / "../../Data/"
        self.clean_data_folder = Path(__file__).parent / "../../CleanData/"
        self.norm_data = Path(__file__).parent / "../../NormalizedData/"
        self.label_file = Path(__file__).parent / "../../labels.xlsx"
        self.labels = []
        self.files_util = File()
        self.labeller = Labeller()

    def process_images(self) -> None:
        """
        Process images prior to normalization and training
        - Renames images
        - Resizes them if necessary
        """

        for folder in os.listdir(self.data_folder):
            # Use absolute path in order to have a valid path to traverse
            abs_folder = os.path.join(self.data_folder, folder)
            valid_extensions = ["HEIC", "jpg", "png", "jpeg"]
            print(f"Processing folder {abs_folder}")

            for index, file in enumerate(os.listdir(abs_folder)):
                abs_file = os.path.join(abs_folder, file)
                ext = file.split(".")[1]
                # Check if the file is mislabelled with the wrong type of extension
                if ext not in valid_extensions:
                    # Assume that the extension has space between it and some other text
                    ext = valid_extensions[3]
                file_name = f"{folder}_{index}.{ext}"
                out_path = os.path.join(self.clean_data_folder, file_name)
                shutil.copy(abs_file, out_path)
                print(f"Saving file {file_name}")
                self.labeller.label_images(folder, file_name, index)

        self.labeller.save_labels()
    
    @staticmethod
    def process_labels(data_folder: str, label_path: str):
        """
        Process the labels and images and normalizes the image arrays
        """
        labels_df = read_excel(label_path)
        labels_df.head()
        image_size = (150, 150)

        x = []  # Array for image
        y = []  # Array of label strings

        for _, row in labels_df.iterrows():
            img_path = os.path.join(data_folder, row["Image"])
            if os.path.isfile (img_path) is False:
                continue
            img = load_img(img_path, target_size=image_size)
            x.append(img_to_array(img))
            y.append(row["Label"])

        X = array(x) / 255.0  # Normalization
        Y = array(y)
        return X, Y

    def resize_all_pictures(self) -> int:
        """
        This function ensures that all files are of the same size
        """
        total_resized_images = 0
        invalid_images = 0
        total_processed_images = 0

        for _, file in enumerate(os.listdir(self.clean_data_folder)):
            try:
                abs_file = os.path.join(self.clean_data_folder, file)
                total_resized_images += resize_picture(abs_file)
                total_processed_images += 1
                print(f"Total images processed: {total_processed_images}")
            except Exception:
                invalid_images += 1
        return invalid_images, total_resized_images
