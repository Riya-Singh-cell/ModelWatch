import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
)


def calculate_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
    }


def calculate_metrics(y_true, y_pred, y_proba=None, classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "roc_auc": None}
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": None,
                "recall": None, "f1_score": None, "roc_auc": None}
    kwargs = {"average": "binary" if len(unique_labels) == 2 else "weighted", "zero_division": 0}
    if len(unique_labels) == 2:
        kwargs["pos_label"] = unique_labels[-1]
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, **kwargs)),
        "f1_score": float(f1_score(y_true, y_pred, **kwargs)),
    }
    try:
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                result["roc_auc"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted", labels=classes
                ))
            elif y_proba.ndim == 2:
                result["roc_auc"] = float(roc_auc_score(
                    y_true, y_proba[:, 1], labels=classes
                ))
            else:
                result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        else:
            result["roc_auc"] = None
    except (ValueError, TypeError):
        result["roc_auc"] = None
    return result
