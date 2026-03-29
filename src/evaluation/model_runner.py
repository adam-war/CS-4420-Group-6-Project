# ============================================
# MODEL RUNNER (FINAL WITH EXPORTS)
# ============================================

from pathlib import Path
from datetime import datetime
import json
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.api.types import is_numeric_dtype

from config.config import (
    TARGET_COLUMN,
    DATASETS_TO_RUN,
    SUBSAMPLE_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
    MODELS_TO_RUN,
    DO_GRID_SEARCH,
    CV_FOLDS,
    SCORING,
    EXPORT_RESULTS,
    SCALING_MODELS,
    BASELINE_PARAMS,
    GRID_PARAMS,
)

from src.data.dataloader import get_features_and_target
from src.evaluation.metrics import (
    evaluate_predictions,
    save_confusion_matrix_figure,
    save_classification_report_text,
    save_metrics_json_like,
)

from src.models.adaline import build_model as build_adaline
from src.models.logistic_regression import build_model as build_logistic_regression
from src.models.svm import build_linear_svm, build_rbf_svm
from src.models.knn import build_model as build_knn
from src.models.decision_tree import build_model as build_decision_tree
from src.models.random_forest import build_model as build_random_forest
from src.models.bagging import build_model as build_bagging
from src.models.sgd_classifier import build_model as build_sgd_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models"


# ============================================
# MODEL FACTORY
# ============================================

def build_estimator(model_name, params):
    if model_name == "adaline":
        return build_adaline(params)
    if model_name == "logistic_regression":
        return build_logistic_regression(params)
    if model_name == "svm_linear":
        return build_linear_svm(params)
    if model_name == "svm_rbf":
        return build_rbf_svm(params)
    if model_name == "knn":
        return build_knn(params)
    if model_name == "decision_tree":
        return build_decision_tree(params)
    if model_name == "random_forest":
        return build_random_forest(params)
    if model_name == "bagging":
        return build_bagging(params)
    if model_name == "sgd_classifier":
        return build_sgd_classifier(params)

    raise ValueError(f"Unknown model: {model_name}")


# ============================================
# PIPELINE
# ============================================

def build_pipeline(model_name, estimator, X_train):
    numeric_features = [c for c in X_train.columns if is_numeric_dtype(X_train[c])]
    categorical_features = [c for c in X_train.columns if not is_numeric_dtype(X_train[c])]

    transformers = []

    if numeric_features:
        if model_name in SCALING_MODELS:
            transformers.append(("num", StandardScaler(), numeric_features))
        else:
            transformers.append(("num", "passthrough", numeric_features))

    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def get_prefixed_param_grid(model_name):
    if model_name not in GRID_PARAMS:
        return None
    return {f"model__{k}": v for k, v in GRID_PARAMS[model_name].items()}


# ============================================
# OUTPUT HELPERS
# ============================================

def create_run_directories():
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_name

    if EXPORT_RESULTS:
        run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, run_name


def create_dataset_directories(run_dir, dataset_name):
    dataset_dir = run_dir / dataset_name
    models_dir = dataset_dir / "individual_models"

    if EXPORT_RESULTS:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

    return dataset_dir, models_dir


def save_dataset_metadata(dataset_dir, dataset_name, feature_names, X, y, X_train, X_test, y_train, y_test):
    metadata = {
        "dataset_name": dataset_name,
        "target_column": TARGET_COLUMN,
        "subsample_size": SUBSAMPLE_SIZE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": STRATIFY,
        "do_grid_search": DO_GRID_SEARCH,
        "cv_folds": CV_FOLDS,
        "scoring": SCORING,
        "models_to_run": MODELS_TO_RUN,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "X_full_shape": list(X.shape),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_full_size": int(len(y)),
        "y_train_size": int(len(y_train)),
        "y_test_size": int(len(y_test)),
        "class_distribution": {
            str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()
        },
    }

    with open(dataset_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    pd.DataFrame({"feature_name": feature_names}).to_csv(
        dataset_dir / "feature_names.csv",
        index=False
    )


# ============================================
# SUBSAMPLE
# ============================================

def apply_optional_subsample(X, y):
    if SUBSAMPLE_SIZE is None or SUBSAMPLE_SIZE >= len(y):
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=SUBSAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if STRATIFY else None,
    )
    return X_sub, y_sub


# ============================================
# TRAIN ONE MODEL
# ============================================

def train_single_model(model_name, dataset_name, X_train, X_test, y_train, y_test, class_names, model_output_dir):
    print("\n==============================")
    print(f"DATASET: {dataset_name}")
    print(f"MODEL:   {model_name}")
    print("==============================")

    estimator = build_estimator(model_name, BASELINE_PARAMS[model_name].copy())
    pipeline = build_pipeline(model_name, estimator, X_train)

    best_params = None
    best_cv_score = None

    start_train = time.time()

    if DO_GRID_SEARCH:
        param_grid = get_prefixed_param_grid(model_name)

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=SCORING,
            cv=CV_FOLDS,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_
    else:
        model = pipeline.fit(X_train, y_train)

    end_train = time.time()

    start_test = time.time()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    end_test = time.time()

    train_metrics, _ = evaluate_predictions(y_train, y_train_pred)
    test_metrics, _ = evaluate_predictions(y_test, y_test_pred)

    metrics_dict = {
        "dataset_name": dataset_name,
        "model": model_name,
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_precision_macro": train_metrics["precision_macro"],
        "test_precision_macro": test_metrics["precision_macro"],
        "train_recall_macro": train_metrics["recall_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "train_f1_macro": train_metrics["f1_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "best_cv_score": best_cv_score,
        "train_time": end_train - start_train,
        "test_time": end_test - start_test,
    }

    print(f"Train Acc: {metrics_dict['train_accuracy']:.4f}")
    print(f"Test  Acc: {metrics_dict['test_accuracy']:.4f}")

    if EXPORT_RESULTS:
        model_output_dir.mkdir(parents=True, exist_ok=True)

        save_metrics_json_like(metrics_dict, model_output_dir / "metrics.csv")

        save_classification_report_text(
            y_train,
            y_train_pred,
            model_output_dir / "classification_report_train.txt"
        )
        save_classification_report_text(
            y_test,
            y_test_pred,
            model_output_dir / "classification_report_test.txt"
        )

        save_confusion_matrix_figure(
            y_train,
            y_train_pred,
            class_names=class_names,
            output_path=model_output_dir / "confusion_matrix_train.png",
            title=f"Train Confusion Matrix - {dataset_name} - {model_name}"
        )
        save_confusion_matrix_figure(
            y_test,
            y_test_pred,
            class_names=class_names,
            output_path=model_output_dir / "confusion_matrix_test.png",
            title=f"Test Confusion Matrix - {dataset_name} - {model_name}"
        )

        if best_params is not None:
            with open(model_output_dir / "best_params.json", "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=4)

        if best_cv_score is not None:
            with open(model_output_dir / "best_cv_score.txt", "w", encoding="utf-8") as f:
                f.write(str(best_cv_score))

    return metrics_dict


# ============================================
# RUN ONE DATASET
# ============================================

def run_single_dataset(dataset_name, run_dir):
    X, y, feature_names = get_features_and_target(dataset_name)
    X, y = apply_optional_subsample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if STRATIFY else None,
    )

    class_names = sorted(pd.Series(y).astype(str).unique().tolist())

    dataset_dir, models_dir = create_dataset_directories(run_dir, dataset_name)

    if EXPORT_RESULTS:
        save_dataset_metadata(
            dataset_dir,
            dataset_name,
            feature_names,
            X,
            y,
            X_train,
            X_test,
            y_train,
            y_test,
        )

    results = []

    for model_name in MODELS_TO_RUN:
        model_output_dir = models_dir / model_name

        result = train_single_model(
            model_name,
            dataset_name,
            X_train,
            X_test,
            y_train,
            y_test,
            class_names,
            model_output_dir,
        )
        results.append(result)

    df = pd.DataFrame(results)

    summary_columns = [
        "dataset_name",
        "model",
        "train_accuracy",
        "test_accuracy",
        "train_precision_macro",
        "test_precision_macro",
        "train_recall_macro",
        "test_recall_macro",
        "train_f1_macro",
        "test_f1_macro",
        "best_cv_score",
        "train_time",
        "test_time",
    ]

    for col in summary_columns:
        if col not in df.columns:
            df[col] = None

    df = df[summary_columns]
    df = df.sort_values(by="test_accuracy", ascending=False)

    print("\n==============================")
    print(f"DATASET SUMMARY: {dataset_name}")
    print("==============================")
    print(df)

    if EXPORT_RESULTS:
        df.to_csv(dataset_dir / "summary.csv", index=False)

    return df


# ============================================
# MAIN RUNNER
# ============================================

def run_all_models():
    run_dir, run_name = create_run_directories()

    all_results = []

    for dataset_name in DATASETS_TO_RUN:
        df = run_single_dataset(dataset_name, run_dir)
        all_results.append(df)

    overall = pd.concat(all_results, ignore_index=True)
    overall = overall.sort_values(by=["dataset_name", "test_accuracy"], ascending=[True, False])

    print("\n==============================")
    print("OVERALL SUMMARY")
    print("==============================")
    print(overall)

    if EXPORT_RESULTS:
        overall.to_csv(run_dir / "overall_summary.csv", index=False)

    return overall, run_dir, run_name