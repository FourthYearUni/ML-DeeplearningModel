"""
This module handles data cleaning and labelling.
@author: Alain Mugisha (U2083264)
"""

import os
import shutil
from pathlib import Path

import cv2
from ..Utils.files import File
from ..Utils.labeller import Labeller


class Cleaner:
    """
    This class handles data cleaning and labelling.
    """

    def __init__(self) -> None:
        self.data_folder = Path(__file__).parent / "../../Data/"
        self.clean_data_folder = Path(__file__).parent / "../../CleanData/"
        self.norm_data = Path(__file__).parent / "../../NormalizedData/"
        self.label_file = Path(__file__).parent / "../../labels.csv"
        self.labels = []
        self.files_util = File()
        self.labeller = Labeller()

    def rename_files(self) -> None:
        """
        Rename files in a folder.
        """
       
        for folder in os.listdir(self.data_folder):
            # Use absolute path in order to have a valid path to traverse
            abs_folder = os.path.join(self.data_folder, folder)
            print(f"Processing folder {abs_folder}")
            
            for index, file in enumerate(os.listdir(abs_folder)):
                abs_file = os.path.join(abs_folder, file)
                ext = file.split(".")[1]
                file_name = f"{folder}_{index}.{ext}"
                out_path = os.path.join(self.clean_data_folder, file_name)
                shutil.copy(abs_file, out_path)
                self.labels.append((folder, file_name))
                print(f"Saving file {file_name}")
                self.labeller.label_images(folder, file_name, index)
                
        self.labeller.save_labels()

    def normalize_images(self) -> None:
        """
        Normalize an image.
        """
        for image in os.listdir(self.clean_data_folder):
            im_path = os.path.join(self.clean_data_folder, image)
            normalized_image = cv2.imread(im_path) / 255.0
            out_path = os.path.join(self.norm_data, image)
            cv2.imwrite(out_path, normalized_image)
            print(f"Saving file {image}")


if __name__ == "__main__":
    cleaner = Cleaner()
    cleaner.rename_files()
    # cleaner.normalize_images()
