"""Paper metrics: macro OVR AUC and RMSE / target standard deviation."""
import numpy as np
from sklearn.metrics import roc_auc_score

def classification_auc(y_true, probabilities, labels):
    label_index = {label: i for i, label in enumerate(labels)}
    encoded = np.asarray([label_index[label] for label in y_true], dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 2:
        return float(roc_auc_score(encoded, probabilities[:, 1]))
    return float(roc_auc_score(encoded, probabilities, labels=np.arange(len(labels)),
                               average="macro", multi_class="ovr"))

def regression_nrmse(y_true, predictions):
    y = np.asarray(y_true, dtype=float)
    prediction = np.asarray(predictions, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite targets and predictions")
    denominator = float(np.std(y))
    if denominator <= 0:
        raise ValueError("NRMSE is undefined for a constant target")
    return float(np.sqrt(np.mean((y - prediction) ** 2)) / denominator)

def evaluate(task, y_true, predictions, labels=None):
    if task == "classification":
        return {"auc": classification_auc(y_true, predictions, labels)}
    return {"nrmse": regression_nrmse(y_true, predictions)}
