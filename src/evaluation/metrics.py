# ============================================
# EVALUATION METRICS
# ============================================

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def compute_basic_metrics(y_true, y_pred):
    """
    Compute main evaluation metrics for multiclass classification.
    """
    accuracy = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": report_dict["macro avg"]["precision"],
        "recall_macro": report_dict["macro avg"]["recall"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
    }

    return metrics, report_dict


def get_classification_report_text(y_true, y_pred):
    """
    Return classification report as formatted text.
    """
    return classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )


def get_confusion_matrix(y_true, y_pred, labels=None):
    """
    Return confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def save_confusion_matrix_figure(y_true, y_pred, class_names, output_path, title=None):
    """
    Save confusion matrix as PNG figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_classification_report_text(y_true, y_pred, output_path):
    """
    Save classification report to a text file.
    """
    report_text = get_classification_report_text(y_true, y_pred)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)


def save_metrics_json_like(metrics_dict, output_path):
    """
    Save a metrics dictionary as a simple CSV with one row.
    """
    df = pd.DataFrame([metrics_dict])
    df.to_csv(output_path, index=False)


def evaluate_predictions(y_true, y_pred):
    """
    Full evaluation wrapper.
    Returns:
        metrics_dict
        report_dict
    """
    metrics_dict, report_dict = compute_basic_metrics(y_true, y_pred)
    return metrics_dict, report_dict


def add_timing_metrics(metrics_dict, train_time, test_time):
    """
    Add timing information to metrics dictionary.
    """
    metrics_dict = metrics_dict.copy()
    metrics_dict["train_time"] = train_time
    metrics_dict["test_time"] = test_time
    return metrics_dict


def time_training_and_prediction(model, X_train, y_train, X_test):
    """
    Fit model and predict while measuring training and inference time.
    Returns:
        fitted_model
        y_pred
        train_time
        test_time
    """
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()

    train_time = end_train - start_train
    test_time = end_test - start_test

    return model, y_pred, train_time, test_time