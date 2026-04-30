# ============================================
# PROJECT CONFIGURATION
# ============================================

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for one processed dataset used by the model runner."""
    name: str
    path: str
    target_column: str


DATASETS_TO_RUN = [
    "manual_imputed",
]

DATASET_CONFIGS = {
    "manual_imputed": DatasetConfig(
        name="manual_imputed",
        path="outputs/preprocessing/manual_imputed/dataset.csv",
        target_column="Alzheimer’s Diagnosis",
    ),
    "manual_dropna": DatasetConfig(
        name="manual_dropna",
        path="outputs/preprocessing/manual_dropna/dataset.csv",
        target_column="Alzheimer’s Diagnosis",
    ),
    "auto_imputed": DatasetConfig(
        name="auto_imputed",
        path="outputs/preprocessing/auto_imputed/dataset.csv",
        target_column="Alzheimer’s Diagnosis",
    ),
    "auto_dropna": DatasetConfig(
        name="auto_dropna",
        path="outputs/preprocessing/auto_dropna/dataset.csv",
        target_column="Alzheimer’s Diagnosis",
    ),

    # Example for another processed dataset:
    # "alzheimers_prediction_manual_imputed": DatasetConfig(
    #     name="alzheimers_prediction_manual_imputed",
    #     path="alzheimers_prediction/outputs/preprocessing/manual_imputed/dataset.csv",
    #     target_column="Diagnosis",
    # ),
}

# Backward-compatible aliases for old imports.
DATASET_PATHS = {name: cfg.path for name, cfg in DATASET_CONFIGS.items()}
TARGET_COLUMN = "Alzheimer’s Diagnosis"
SUBSAMPLE_SIZE = None
TEST_SIZE = 0.2
RANDOM_STATE = 1
STRATIFY = True

MODELS_TO_RUN = [
    "logistic_regression",
    "svm_linear",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
]

DO_GRID_SEARCH = False
CV_FOLDS = 3
SCORING = "accuracy"
EXPORT_RESULTS = True

SCALING_MODELS = [
    "adaline",
    "logistic_regression",
    "svm_linear",
    "svm_rbf",
    "knn",
    "sgd_classifier",
]

BASELINE_PARAMS = {
    "adaline": {"eta": 0.01, "n_iter": 100, "random_state": RANDOM_STATE},

    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
    },

    "svm_linear": {
        "C": 1.0,
        "kernel": "linear",
        "probability": True,
        "random_state": RANDOM_STATE,
    },

    "svm_rbf": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "random_state": RANDOM_STATE,
    },

    "knn": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "p": 2},

    "decision_tree": {
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": RANDOM_STATE,
    },

    "random_forest": {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },

    "bagging": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},

    "sgd_classifier": {
        "loss": "log_loss",
        "alpha": 0.0001,
        "max_iter": 1000,
        "tol": 1e-3,
        "random_state": RANDOM_STATE,
    },

    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "random_state": RANDOM_STATE,
    },
}

GRID_PARAMS = {
    "adaline": {"eta": [0.0001, 0.001, 0.01, 0.1], "n_iter": [50, 100, 200]},
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },
    "svm_linear": {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "svm_rbf": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "gamma": [0.001, 0.01, 0.1, 1.0, "scale"],
    },
    "knn": {"n_neighbors": [3, 5, 7, 9, 11], "p": [1, 2], "weights": ["uniform", "distance"]},
    "decision_tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "bagging": {"n_estimators": [50, 100, 200]},
    "sgd_classifier": {
        "loss": ["hinge", "log_loss"],
        "alpha": [0.00001, 0.0001, 0.001, 0.01],
        "max_iter": [1000],
    },
    "gradient_boosting": {
        "n_estimators": [150, 200, 250],
        "learning_rate": [0.05, 0.01, 0.005],
        "max_depth": [2, 3, 4],
    },
}
