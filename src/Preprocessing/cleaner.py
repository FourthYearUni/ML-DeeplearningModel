"""
This module handles data cleaning and labelling.
@author: Alain Mugisha (U2083264)
"""

import os
import shutil
from pathlib import Path
import random
from os.path import join

from pandas import read_excel
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import array
from PIL import UnidentifiedImageError


from src.Utils.files import File
from src.Utils.labeller import Labeller
from src.Utils.resize import resize_picture


class Cleaner:
    """
    This class handles data cleaning and labelling.
    """

    def __init__(self) -> None:
        self.data_folder = Path(__file__).parent / "../../Data/"
        self.clean_data_folder = Path(__file__).parent / "../../CleanData/"
        self.norm_data = Path(__file__).parent / "../../NormalizedData/"
        self.label_file = Path(__file__).parent / "../../labels.xlsx"
        self.sampled_data_folder = Path(__file__).parent / "../../SampledData"
        self.labels = []
        self.files_util = File()
        self.labeller = Labeller()

    def process_images(self) -> None:
        """
        Process images prior to normalization and training
        - Renames images
        - Resizes them if necessary
        """

        for folder in os.listdir(self.sampled_data_folder):
            # Use absolute path in order to have a valid path to traverse
            abs_folder = os.path.join(self.sampled_data_folder, folder)
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
                self.labeller.label_images(folder, file_name)

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
            if os.path.isfile(img_path) is False:
                continue
            # Attempt to load the image and if not skip it.
            try:
                img = load_img(img_path, target_size=image_size)
            except UnidentifiedImageError:
                print(f"{img_path} is invalid skipping...")
                continue
            x.append(img_to_array(img))
            y.append(row["Label"])

        X = array(x) / 255.0  # Normalization
        Y = array(y)
        return X, Y

    def resize_all_pictures(self) -> tuple[int, int]:
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

    def sampler(self) -> None:
        """
        Deletes files that don't make the sample.
        """
        sample_rate = 5000 / 12000

        for folder in os.listdir(self.data_folder):
            #  Get the number of files and the absolute path of the current child folder
            abs_path = os.path.join(self.data_folder, folder)
            total_number_files = os.listdir(abs_path).__len__()
            files_to_save = int(total_number_files * sample_rate)
            file_indexes_to_save = random.sample(
                range(0, total_number_files), k=files_to_save
            )
            os.makedirs(f"{self.sampled_data_folder}/{folder}", exist_ok=True)
            for index, file in enumerate(os.listdir(abs_path)):
                file_abs_path = os.path.join(abs_path, file)
                if index in file_indexes_to_save:
                    new_folder = join(self.sampled_data_folder, folder)
                    shutil.copy(file_abs_path, new_folder)
