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
from ..Utils.labeller import Labeller
from ..Utils.resize import resize_picture


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

    def process_images(self) -> None:
        """
        Process images prior to normalization and training
        - Renames images
        - Resizes them if necessary
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
                resize_picture(out_path)
                print(f"Saving file {file_name}")
                self.labeller.label_images(folder, file_name, index)
                
        self.labeller.save_labels()
    
    def process_labels(data_folder: str, label_path: str):
        """
        Process the labels and images and normalizes the image arrays
        """
        labelsDf = read_excel(label_path)
        labelsDf.head()
        image_size = (150, 150)
        
        x = [] # Array for image
        y = [] # Array of label strings

        for _, row in labelsDf.iterrows():
            img_path = os.path.join(data_folder, row["Image"])
            img = load_img(img_path, target_size=image_size)
            x.append(img_to_array(img))
            y.append(row['Label'])
        
        X = array(x) / 255.0 # Normalization
        Y = array(y)




if __name__ == "__main__":
    cleaner = Cleaner()
    cleaner.rename_files()
    # cleaner.normalize_images()
