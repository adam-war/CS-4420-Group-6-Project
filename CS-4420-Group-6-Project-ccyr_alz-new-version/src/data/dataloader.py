# ============================================
# DATA LOADER
# ============================================

from pathlib import Path
import pandas as pd

from config.config import DATASETS_TO_RUN, DATASET_PATHS, TARGET_COLUMN


# --------------------------------------------
# 1. PROJECT ROOT
# --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------
# 2. LOAD DATASET PATH
# --------------------------------------------

def get_dataset_path(dataset_name):
    """
    Return the absolute path of the selected processed dataset.
    """
    if dataset_name not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. "
            f"Available options are: {list(DATASET_PATHS.keys())}"
        )

    relative_path = DATASET_PATHS[dataset_name]
    dataset_path = PROJECT_ROOT / relative_path

    return dataset_path


# --------------------------------------------
# 3. LOAD DATAFRAME
# --------------------------------------------

def load_dataset(dataset_name):
    """
    Load the selected dataset as a pandas DataFrame.
    """
    dataset_path = get_dataset_path(dataset_name)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            "Make sure preprocessing.py has already created this dataset."
        )

    df = pd.read_csv(dataset_path)
    return df


# --------------------------------------------
# 4. SPLIT FEATURES AND TARGET
# --------------------------------------------

def get_features_and_target(dataset_name, target_column=TARGET_COLUMN):
    """
    Load dataset and split it into:
    - X: feature DataFrame
    - y: target Series
    - feature_names: list of feature names
    """
    df = load_dataset(dataset_name)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    feature_names = [col for col in df.columns if col != target_column]

    X = df[feature_names].copy()
    y = df[target_column].copy()

    return X, y, feature_names


# --------------------------------------------
# 5. PRINT DATASET INFO
# --------------------------------------------

def print_dataset_info(dataset_name, target_column=TARGET_COLUMN):
    """
    Print a dataset summary for validation.
    """
    df = load_dataset(dataset_name)
    feature_names = [col for col in df.columns if col != target_column]

    print("\n==============================")
    print("DATA LOADER SUMMARY")
    print("==============================")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset path: {get_dataset_path(dataset_name)}")
    print(f"Shape: {df.shape}")
    print(f"Target column: {target_column}")
    print(f"Number of input features: {len(feature_names)}")

    print("\nInput features:")
    for feat in feature_names:
        print(f"- {feat}")

    print("\nTarget distribution:")
    print(df[target_column].value_counts())


# --------------------------------------------
# 6. MAIN (OPTIONAL TEST)
# --------------------------------------------

if __name__ == "__main__":
    print("\n==============================")
    print("TESTING ALL DATASETS IN DATASETS_TO_RUN")
    print("==============================")

    for dataset_name in DATASETS_TO_RUN:
        print_dataset_info(dataset_name)