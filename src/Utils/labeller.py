"""
This module handles the labelling process of the images
"""
from pathlib import Path
from openpyxl import Workbook


class Labeller:
    """
    Provides methods for labelling images
    """

    def __init__(self) -> None:
        self.label_file = Path(__file__).parent / "../../labels.xlsx"
        self.workbook = Workbook()
        self.sheet = self.workbook.active
        self.sheet["A1"] = "Label"
        self.sheet["B1"] = "Image"
    
    def label_images(self, label: str, image: str, index: int) -> None:
        """
        Labels the images
        """
        print(f"Index is {index}")
        self.sheet.append([label, image])
    
    def save_labels(self) -> None:
        """
        Saves the labels to a file
        """
        self.workbook.save(self.label_file)
        print("Labels saved successfully")
    





        