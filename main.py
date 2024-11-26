"""
Main entry point for the application
"""

from src.Preprocessing.cleaner import Cleaner
from src.Training.trainer import Trainer


class Main:
    """
    Main class fpr the application
    """

    def __init__(self) -> None:
        self.cleaner = Cleaner()
        image_array, label_array = self.cleaner.process_labels(
            self.cleaner.clean_data_folder, self.cleaner.label_file
        )
        self.trainer = Trainer(labels=label_array, images=image_array)

    def run(self) -> None:
        """
        Application entry point
        """
        # rename files and move them to the clean folder
        # self.cleaner.process_images()

        # resize all pictures
        # total_invalid, total_resized = self.cleaner.resize_all_pictures()
        # print(
        #     f"Number of invalid pictures: {total_invalid}\nNumber of resized images{total_resized}"
        # )

        # Train CNN model
        predictions = self.trainer.build_cnn_model()
        print(predictions)


if __name__ == "__main__":
    main = Main()
    main.run()
