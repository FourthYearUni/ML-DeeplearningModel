"""
Main entry point for the application
"""

from src.Preprocessing.cleaner import Cleaner
from src.Training.trainer import Trainer
from src.PostTraining.label_issues import LabelIssues
from src.Utils.files import File


class Main:
    """
    Main class fpr the application
    """

    def __init__(self) -> None:
        self.cleaner = Cleaner()
        self.files = File()
        self.label_issues = LabelIssues()

        # Cleanup files generated by the previous run
        self.files.delete_folder(self.cleaner.clean_data_folder)
        self.files.delete_folder(self.cleaner.sampled_data_folder)
        self.files.delete_folder(self.cleaner.problematic_images_folder)
        self.files.delete_folder(self.cleaner.proper_images_folder)

        # Sample the data
        self.cleaner.sampler()

        # rename files and move them        to the clean folder
        self.cleaner.process_images(self.cleaner.sampled_data_folder)

    def run(self) -> None:
        """
        Application entry point
        """
        image_array, label_array = self.cleaner.process_labels(
            self.cleaner.clean_data_folder, self.cleaner.label_file
        )
        print("================= Finished Cleaning and Preprocessing ====================")
        self.trainer = Trainer(labels=label_array, images=image_array)

        # resize all pictures
        total_invalid, total_resized = self.cleaner.resize_all_pictures()
        print(
            f"Number of invalid pictures: {total_invalid}\nNumber of resized images{total_resized}"
        )
        # Encode labels
        self.trainer.encode_categorical()

        # Call the splitter to obtain test and training sets
        x_train, x_test, y_train, y_test, x_val, y_val = self.trainer.split()

        # Train CNN model
        print("================= Starting Training =================")
        predictions = self.trainer.build_cnn_model(x_train, x_test, y_train, y_test)
        print(predictions)
        # Indentify and report label errors
        print("================  Post Training =====================")
        self.label_issues.find_and_fix_label_issues(y_train, predictions, x_train)
        self.label_issues.report_errors()
        # _ = self.trainer.build_cnn_model(n_xtrain, x_test, n_ytrain, y_test)


if __name__ == "__main__":
    main = Main()
    main.run()
