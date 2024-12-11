"""
Main entry point for the application
"""
from cleanlab.filter import find_label_issues
from numpy import argmax, delete, ndarray
from os import path
from pathlib import Path

from src.Preprocessing.cleaner import Cleaner
from src.Training.trainer import Trainer
from src.Utils.files import File


class Main:
    """
    Main class fpr the application
    """

    def __init__(self) -> None:
        self.cleaner = Cleaner()
        # Sample the data
        # self.cleaner.sampler()

        # rename files and move them to the clean folder
        self.cleaner.process_images()
        self.files = File()
    
    def find_and_fix_label_issues(self, y_train, predictions, x_train) -> tuple[ndarray, ndarray]:
        """
        This method uses cleanlab to do dataset clean up
        - Finds label issues
        - Removes mislabelled images in x_train
        """
        # Use Cleanlab to find the label issues
        # Flatten y_train to 1-D
        flat_y_train = argmax(y_train, axis=1)
        label_issues = find_label_issues(labels=flat_y_train, pred_probs=predictions)
        flat_pred = argmax(predictions, axis=1)

        for x, issue in enumerate(label_issues):
            with open('label_issues.txt', 'a+') as f:
                if flat_y_train[x] != flat_pred[x]:
                    img = self.files.construct_image(x_train[x])
                    f.write(f"Issue: Pred: {flat_pred[x]} Actual: {flat_y_train[x]}\n")
                    file_name = f"Pred_{flat_pred[x]}_Actual_{flat_y_train[x]}.jpeg"
                    full_path = path.join(Path(__file__).parent / "problematic_images", file_name)
                    img.save(full_path)
                    delete(x_train, x, axis=0)
                    delete(y_train, x)
        return x_train, y_train


    
    def run(self) -> None:
        """
        Application entry point
        """

        image_array, label_array = self.cleaner.process_labels(
            self.cleaner.clean_data_folder, self.cleaner.label_file
        )
        print(f"The shape of the labells is {label_array}")
        self.trainer = Trainer(labels=label_array, images=image_array)

        # resize all pictures
        total_invalid, total_resized = self.cleaner.resize_all_pictures()
        print(
            f"Number of invalid pictures: {total_invalid}\nNumber of resized images{total_resized}"
        )
        # Encode labels
        self.trainer.encode_categorical()

        # Call the splitter to obtain test and training sets
        x_train, x_test, y_train, y_test = self.trainer.split()
        print(x_train)
        # Train CNN model
        predictions = self.trainer.build_cnn_model(x_train, x_test, y_train, y_test)
        
        # Retrain the model with new values
        n_xtrain, n_ytrain = self.find_and_fix_label_issues(y_train, predictions,x_train)
        _ = self.trainer.build_cnn_model(n_xtrain, x_test, n_ytrain, y_test)

if __name__ == "__main__":
    main = Main()
    main.run()
