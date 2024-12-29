"""
@author: Alain Christian Mugisha
@brief: This module provides reporting capabilities. The intention is to have a central
module in charge of parsing, processing large logfiles and extract insights about the model
and it's performance and architecture.
"""
from typing import List
from pandas import DataFrame, read_json
import matplotlib.pyplot as pylt
from random import random
from pathlib import Path

class Reporting:
    """
    This class provides methods for loading a dataset and plotting insights
    """

    def __init__(self):
        """
        Base constructor
        """
        self.dataset = None
    
    def load_from_file(self, path: Path) -> DataFrame:
        """
        This function will load data from a file and return a dataframe.
        TODO: Modify the function to accomodate other file types that aren't JSON
        """
        dataset = read_json(path)
        return dataset

    def bar_plot(self, properties: List[str], values: list, title: str, x_label:str):
        """
        This function will plot the provided dataset using the precised properties
        """
        fig, ax = pylt.subplots(figsize=(10,8))
        
        labels = properties
        colors = [(random(), random(), random(), 1) for value in values]
        ax.bar(properties, values, label=labels, color=colors)
        ax.set_xticklabels(properties, rotation=-20)
        ax.legend(title=title)
        ax.set_ylabel("Error Count")
        ax.set_xlabel(x_label)
        
        return pylt


