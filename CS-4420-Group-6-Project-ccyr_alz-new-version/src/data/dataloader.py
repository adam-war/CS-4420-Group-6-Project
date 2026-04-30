# ============================================
# DATA LOADER
# ============================================

from pathlib import Path
import pandas as pd

from config.config import DATASETS_TO_RUN, DATASET_CONFIGS


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_dataset_config(dataset_name):
    """Return the DatasetConfig object for a dataset name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. "
            f"Available options are: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def get_dataset_path(dataset_name):
    """Return the absolute path of the selected processed dataset."""
    return PROJECT_ROOT / get_dataset_config(dataset_name).path


def get_target_column(dataset_name):
    """Return the target column for the selected processed dataset."""
    return get_dataset_config(dataset_name).target_column


def load_dataset(dataset_name):
    """Load the selected dataset as a pandas DataFrame."""
    dataset_path = get_dataset_path(dataset_name)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            "Make sure preprocessing has already created this dataset, "
            "or check the path in DATASET_CONFIGS."
        )

    return pd.read_csv(dataset_path)


def get_features_and_target(dataset_name):
    """
    Load dataset and split it into X, y, feature_names, and target_column.
    """
    df = load_dataset(dataset_name)
    target_column = get_target_column(dataset_name)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset '{dataset_name}'. "
            f"Available columns: {list(df.columns)}"
        )

    feature_names = [col for col in df.columns if col != target_column]
    X = df[feature_names].copy()
    y = df[target_column].copy()

    return X, y, feature_names, target_column


def print_dataset_info(dataset_name):
    """Print a dataset summary for validation."""
    df = load_dataset(dataset_name)
    target_column = get_target_column(dataset_name)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset '{dataset_name}'. "
            f"Available columns: {list(df.columns)}"
        )

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


if __name__ == "__main__":
    print("\n==============================")
    print("TESTING ALL DATASETS IN DATASETS_TO_RUN")
    print("==============================")

    for dataset_name in DATASETS_TO_RUN:
        print_dataset_info(dataset_name)
