import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Evaluate results
def evaluate(y_true, y_pred, y_score=None) -> dict[str, float]:

    # Get values from the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Get other metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Store them all in a dictionary
    metrics = {}
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    metrics["accuracy"] = accuracy
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1

    # Handle missing data
    if y_score is None:
        metrics["roc_auc"] = float("nan")
    else:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


# Get relevant data for metrics like roc_auc
def positive_scores(model, X):
    # Call predict proba if it exists in the model
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs[:, 1]

    # Call decision function if it exists in the model
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)

    return None
