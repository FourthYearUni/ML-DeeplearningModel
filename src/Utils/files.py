"""
This module handles any file operations
"""

import os
import hashlib
from os.path import isdir
import subprocess
import json
from pathlib import Path

from openpyxl import Workbook
import pandas as pd
from PIL import Image

class File:
    """
    This class defines methods that do regular file operations some of these include
    - Reading files
    - Writing files
    - Calculating file hash
    - Deleting files
    """

    def __init__(self) -> None:
        self.hash_function = hashlib.sha256()
        self.chunk_size = 4096
        self.error_log = Path(__file__).parent / "../../error_log.json" 

    def calculate_hash(self, file_path: str) -> str:
        """
        Calculate the hash of a file
        """
        result = subprocess.run(["sha256sum", file_path], stdout=subprocess.PIPE)
        return result.stdout.decode("utf-8").split(" ")[0]

    def find_duplicate_files(self, folder_path: str) -> list:
        """
        Find duplicate files in a folder
        """
        hashes = {}
        duplicates = []
        workbook = Workbook()
        sheet = workbook.active
        sheet["A1"] = "Hash"
        sheet["B1"] = "File Path"

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            file_hash = self.calculate_hash(file_path)
            if file_hash in hashes:
                # If the file hash already exists in the dictionary append the file path property
                hashes[file_hash].append(file_path)
                sheet.append([file_hash, file_path])
            else:
                # If the file hash does not exist in the dictionary add it as a new key
                hashes[file_hash] = [file_path]
        workbook.save("hashes.xlsx")

        for files in hashes.values():
            if len(files) > 1:
                duplicates.append(files)

        return duplicates

    def delete_file(self, file_path: str) -> None:
        """
        Delete a file
        """
        os.remove(file_path)
        print(f"Deleted file {file_path}")
    
    def construct_image(self, image_array) -> Image:
        """
        Constructs an Image from a PIL image array
        """
        img = Image.fromarray((image_array * 255).astype('uint8'))
        return img
    
    def save_file(self, dest, img) -> bool:
        """
        Saves a passed img to the passed destination
        """
        
        pass

    def delete_folder(self, folder_path):
        """
        Deletes folders passed on
        """
        for node in os.listdir(folder_path):
            path = os.path.join(folder_path, node)
            if isdir(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    self.delete_file(file_path)
            else:
                self.delete_file(path)
        print(f"Deleted a folder at {folder_path}")
    
    
    def append_to_json(self, file_path: str, contents: dict):
        """
        This utility function will open a file append its contents
        with the contents passed on here.
        """
        with open(file_path, 'r+') as f:
            file_obj = json.load(f)
            file_obj.append(contents)
            f.seek(0)
            json.dump(file_obj, f)

    def report_error(self, img_path, error_type, step, stage):
        """
        This function reports errors that happened during processing of images
        to a json log file using a utility file inside the Files util folder
        """
        error_obj = {
                "img_path": img_path,
                "error_type": error_type,
                "step": step,
                "stage": stage
        }
        self.append_to_json(str(self.error_log), error_obj)
    
