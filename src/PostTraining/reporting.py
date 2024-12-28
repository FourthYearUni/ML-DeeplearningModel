"""
@author: Alain Christian Mugisha
@brief: This module provides reporting capabilities. The intention is to have a central
module in charge of parsing, processing large logfiles and extract insights about the model
and it's performance and architecture.
"""
import pandas as pd


class Reporting:
    """
    This class provides methods for loading a dataset and plotting insights
    """

    def __init__(self):
        """
        Base constructor
        """
