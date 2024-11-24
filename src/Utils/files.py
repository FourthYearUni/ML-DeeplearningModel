"""
This module handles any file operations
"""

import os
import hashlib
import subprocess

from openpyxl import Workbook
import pandas as pd

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
