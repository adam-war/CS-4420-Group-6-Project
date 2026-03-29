# ============================================
# PROJECT CONFIGURATION
# ============================================

# --------------------------------------------
# 1. DATASET SETTINGS
# --------------------------------------------

TARGET_COLUMN = "DX"

DATASETS_TO_RUN = [
    "manual_imputed",
    "manual_dropna",
    "auto_imputed",
    "auto_dropna",
]
# Options:
# "manual_imputed"
# "manual_dropna"
# "auto_imputed"
# "auto_dropna"

SUBSAMPLE_SIZE = None
TEST_SIZE = 0.2
RANDOM_STATE = 1
STRATIFY = True

# --------------------------------------------
# 1B. DATASET PATHS
# --------------------------------------------

DATASET_PATHS = {
    "manual_imputed": "outputs/preprocessing/manual_imputed/dataset.csv",
    "manual_dropna": "outputs/preprocessing/manual_dropna/dataset.csv",
    "auto_imputed": "outputs/preprocessing/auto_imputed/dataset.csv",
    "auto_dropna": "outputs/preprocessing/auto_dropna/dataset.csv",
}


# --------------------------------------------
# 2. MODEL SELECTION
# --------------------------------------------

MODELS_TO_RUN = [
    "adaline",
    "logistic_regression",
    "svm_linear",
    "svm_rbf",
    "knn",
    "decision_tree",
    "random_forest",
    "bagging",
    "sgd_classifier",
]
#options:
#    "adaline"
#    "logistic_regression"
#    "svm_linear"
#    "svm_rbf"
#    "knn"
#    "decision_tree"
#    "random_forest"
#    "bagging"
#    "sgd_classifier"


# --------------------------------------------
# 3. CROSS-VALIDATION AND GRID SEARCH
# --------------------------------------------

DO_GRID_SEARCH = True
CV_FOLDS = 5
SCORING = "accuracy"


# --------------------------------------------
# 4. OUTPUT SETTINGS
# --------------------------------------------

EXPORT_RESULTS = True


# --------------------------------------------
# 5. SCALING SETTINGS
# --------------------------------------------

# Models that REQUIRE feature scaling
SCALING_MODELS = [
    "adaline",
    "logistic_regression",
    "svm_linear",
    "svm_rbf",
    "knn",
    "sgd_classifier",
]


# --------------------------------------------
# 6. BASELINE MODEL PARAMETERS
# --------------------------------------------

BASELINE_PARAMS = {
    "adaline": {
        "eta": 0.01,
        "n_iter": 100,
        "random_state": RANDOM_STATE,
    },

    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "multi_class": "auto",
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

    "knn": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
        "p": 2,
    },

    "decision_tree": {
        "criterion": "gini",
        "max_depth": None,
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

    "bagging": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },

    "sgd_classifier": {
    "loss": "log",
    "alpha": 0.0001,
    "max_iter": 1000,
    "tol": 1e-3,
    "random_state": RANDOM_STATE,
    },
}


# --------------------------------------------
# 7. GRID SEARCH PARAMETER GRIDS
# --------------------------------------------

GRID_PARAMS = {
    "adaline": {
        "eta": [0.0001, 0.001, 0.01, 0.1],
        "n_iter": [50, 100, 200],
    },

    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },

    "svm_linear": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },

    "svm_rbf": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "gamma": [0.001, 0.01, 0.1, 1.0, "scale"],
    },

    "knn": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "p": [1, 2],
        "weights": ["uniform", "distance"],
    },

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

    "bagging": {
        "n_estimators": [50, 100, 200],
    },

    "sgd_classifier": {
        "loss": ["hinge", "log"],
        "alpha": [0.00001, 0.0001, 0.001, 0.01],
        "max_iter": [1000],
    },
}