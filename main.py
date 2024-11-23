"""
Main entry point for the application
"""
from src.Preprocessing.cleaner import Cleaner

class Main:
    """
    Main class fpr the application
    """
    def __init__(self) -> None:
        self.cleaner = Cleaner()

    def run(self) -> None:
        """
        Application entry point
        """
        self.cleaner.rename_files()




if __name__ == "__main__":
    main = Main()
    main.run()