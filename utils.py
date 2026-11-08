# utils.py
# Small helper file used by both train_models.py and app.py.
# Keeping the metric calculation in ONE place means the training script and the
# web app always calculate the scores in exactly the same way.

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# Name of the column we want to predict (the obesity level)
TARGET = "NObeyesdad"

# Each model name and the file it is saved under inside the model/ folder
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Calculate the 6 evaluation scores asked in the assignment.

    y_true    -> the real answers
    y_pred    -> what the model predicted
    y_proba   -> how confident the model was for each class (needed for AUC)
    n_classes -> number of obesity categories (7 here)
    """

    # AUC normally works for 2 classes. We have 7, so we use "one-vs-rest":
    # compare each class against all the others and take the average.
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                            average="macro", labels=list(range(n_classes)))
    except ValueError:
        # This happens if the uploaded file is missing some classes
        auc = float("nan")

    # For precision / recall / F1 with many classes we use "macro" average,
    # which treats every class equally.
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
