"""
@author: Alain Christian Mugisha (avici1)
@brief: Provides methods to find, report and handle label issues
"""

from os import path, makedirs
from numpy import argmax, ndarray, max as max_np
from cleanlab.filter import find_label_issues, confusion_matrix
from pathlib import Path

from pandas import Series

from src.Utils.files import File
from src.Preprocessing.cleaner import Cleaner
from src.PostTraining.reporting import Reporting

class LabelIssues:
    """
    This class provides methods to find, report and handle label issues
    """

    def __init__(self):
        """
        Base constructor
        """
        self.files = File()
        self.cleaner = Cleaner()
        self.reporting = Reporting()

    @staticmethod
    def categorise_errors(prediction: int, confidence_score: float, label: int) -> str:
        """
        This function will categorise errors based on the predictions
        """
        error_type = ""
        if label != prediction:
            if confidence_score > 0.8:
                error_type = "Mislabelled"
            elif 0.4 < confidence_score <= 0.8:
                error_type = "Ambiguous"
            elif confidence_score <= 0.4:
                error_type = "Noise/Outlier"

        return error_type

    def find_and_fix_label_issues(self, y_train, predictions, x_train) -> None:
        """
        This method uses cleanlab to do dataset clean up
        - Finds label issues
        - Removes mislabelled images in x_train
        """
        #: Use Cleanlab to find the label issues
        # Flatten y_train to 1-D
        flat_y_train = argmax(y_train, axis=1)
        label_issues = find_label_issues(labels=flat_y_train, pred_probs=predictions)
        flat_pred = argmax(predictions, axis=1)
        confidence_scores = max_np(predictions, axis=1)

        print(f"Label issues found is {len(label_issues)}")
        for x, issue in enumerate(label_issues):
            with open("label_issues.txt", "a+") as f:
                img = self.files.construct_image(x_train[x])
                f.write(f"Issue: Pred: {flat_pred[x]} Actual: {flat_y_train[x]}\n")
                file_name = f"{x}_Pred_{flat_pred[x]}_Actual_{flat_y_train[x]}.jpeg"
                issue_state = flat_y_train[x] == flat_pred[x]
                folder = ""
                if issue_state == False:
                    folder = f"ProblematicImages/Stage{flat_y_train[x] + 1}"
                else:
                    folder = f"ProperImages/Stage{flat_y_train[x] + 1}"
                makedirs(folder, exist_ok=True)
                full_path = path.join(
                    Path(__file__).parent / f"../../{folder}", file_name
                )
                img.save(full_path)
                if issue_state == False:
                    error_type = self.categorise_errors(
                        flat_pred[x], confidence_scores[x], flat_y_train[x]
                    )
                    self.files.report_error(
                        full_path, error_type, "Post training", str(flat_y_train[x] + 1)
                    )


    def report_errors(self):
        """
        This function will plot insights about errors.
        """
        df = self.reporting.load_from_file(self.files.error_log)
        df["stage"] = df["stage"].apply(lambda x: f"Stage {x}")
        error_count_by_type = df.groupby("error_type")["error_type"].count()
        error_count_by_stage = df.groupby("stage")["stage"].count()
        error_types_list = df["error_type"].unique().tolist()
        error_stages_list = df["stage"].unique().tolist()
        
        # Plotting errors by Type
        properties = error_types_list
        self.reporting.bar_plot(
                properties=properties,
                values=error_count_by_type.values.tolist(),
                title="Errors by type",
                x_label="Error types"
        )

        # Plotting errors by Stage
        properties = error_stages_list
        pylt = self.reporting.bar_plot(
                properties=properties, 
                values=error_count_by_stage.values.tolist(),
                title="Errors by Stage",
                x_label="Stages"
        )

        pylt.show()



