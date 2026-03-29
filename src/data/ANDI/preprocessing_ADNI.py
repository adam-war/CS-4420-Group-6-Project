# ============================================
# FINAL PREPROCESSING SCRIPT
# ============================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# --------------------------------------------
# 0. PATHS
# --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_check" / "clean_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessing"

MANUAL_IMPUTED_DIR = OUTPUT_DIR / "manual_imputed"
MANUAL_DROPNA_DIR = OUTPUT_DIR / "manual_dropna"
AUTO_IMPUTED_DIR = OUTPUT_DIR / "auto_imputed"
AUTO_DROPNA_DIR = OUTPUT_DIR / "auto_dropna"

INPUT_FILENAME = "clean_dataset_DX.csv"
TARGET_COL = "DX"


# --------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------

MANUAL_FEATURES = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "APOE4",
    "MMSE",
    "ADAS13",
    "FAQ",
    "Hippocampus",
    "Ventricles",
    "WholeBrain",
]

UNNECESSARY_COLUMNS = {
    "RID",
    "PTID",
    "VISCODE",
    "EXAMDATE",
    "EXAMDATE_bl",
    "COLPROT",
    "ORIGPROT",
    "SITE",
    "IMAGEUID",
    "IMAGEUID_bl",
    "FSVERSION",
    "FSVERSION_bl",
    "FLDSTRENG",
    "FLDSTRENG_bl",
    "update_stamp",
    "Years_bl",
    "Month_bl",
    "Month",
    "M",
    "DX_bl",
}

AUTO_MAX_MISSING_PERCENT = 10.0


# --------------------------------------------
# 2. HELPERS
# --------------------------------------------

def ensure_output_dirs():
    MANUAL_IMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_DROPNA_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_IMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_DROPNA_DIR.mkdir(parents=True, exist_ok=True)


def load_input_dataset():
    file_path = INPUT_DIR / INPUT_FILENAME
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {file_path}\n"
            f"Make sure dataset_check already created {INPUT_FILENAME}."
        )
    return pd.read_csv(file_path)


def compute_missing_summary(df):
    missing_count = df.isnull().sum()
    missing_percent = 100.0 * missing_count / len(df)

    summary_df = pd.DataFrame({
        "feature": df.columns,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values,
    }).sort_values(by=["missing_percent", "feature"], ascending=[False, True])

    return summary_df


def split_feature_types(df, target_col):
    feature_df = df.drop(columns=[target_col], errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def impute_dataset(df, target_col):
    df_out = df.copy()

    feature_cols = [col for col in df_out.columns if col != target_col]

    numeric_cols = df_out[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_out[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="mean")
        df_out[numeric_cols] = num_imputer.fit_transform(df_out[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df_out[categorical_cols] = cat_imputer.fit_transform(df_out[categorical_cols])

    return df_out


def print_preprocessing_diagnostics(df_before, mode_name):
    print("\n==============================")
    print(f"{mode_name.upper()} DATA BEFORE IMPUTATION")
    print("==============================")

    n_rows = df_before.shape[0]
    rows_with_nan = df_before.isnull().any(axis=1).sum()
    rows_clean = n_rows - rows_with_nan
    df_drop = df_before.dropna()

    print(f"Total rows: {n_rows}")
    print(f"Rows with missing values: {rows_with_nan}")
    print(f"Rows without missing values: {rows_clean}")
    print(f"Rows after dropping NaNs: {df_drop.shape[0]}")
    print(f"Percentage of data kept after drop: {100 * df_drop.shape[0] / n_rows:.2f}%")

    print("\nMissing percentage per feature in this subset:")
    print((df_before.isnull().mean() * 100).sort_values(ascending=False))


def save_outputs(df_before, df_after, target_col, selected_features, out_dir, mode_name, strategy_name):
    missing_before_df = compute_missing_summary(df_before)
    missing_after_df = compute_missing_summary(df_after)

    missing_before_df.to_csv(out_dir / "missing_before.csv", index=False)
    missing_after_df.to_csv(out_dir / "missing_after.csv", index=False)

    feature_df = pd.DataFrame({"feature": selected_features})
    feature_df.to_csv(out_dir / "selected_features.csv", index=False)

    preview_df = df_after.head(10)
    preview_df.to_csv(out_dir / "dataset_preview.csv", index=False)

    target_counts = df_after[target_col].value_counts(dropna=False)
    target_percent = 100.0 * target_counts / len(df_after)

    target_dist_df = pd.DataFrame({
        target_col: target_counts.index.astype(str),
        "count": target_counts.values,
        "percentage": target_percent.values,
    })
    target_dist_df.to_csv(out_dir / "target_distribution.csv", index=False)

    numeric_cols, categorical_cols = split_feature_types(df_after, target_col)

    summary = {
        "mode": mode_name,
        "strategy": strategy_name,
        "target": target_col,
        "rows": int(df_after.shape[0]),
        "total_columns": int(df_after.shape[1]),
        "number_of_features": int(df_after.shape[1] - 1),
        "numeric_features": int(len(numeric_cols)),
        "categorical_features": int(len(categorical_cols)),
        "selected_features": selected_features,
        "missing_values_before": int(df_before.isnull().sum().sum()),
        "missing_values_after": int(df_after.isnull().sum().sum()),
        "rows_before": int(df_before.shape[0]),
        "rows_after": int(df_after.shape[0]),
        "class_distribution": {str(k): int(v) for k, v in target_counts.to_dict().items()},
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Mode: {mode_name}\n")
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Target: {target_col}\n")
        f.write(f"Rows: {df_after.shape[0]}\n")
        f.write(f"Total columns: {df_after.shape[1]}\n")
        f.write(f"Number of features: {df_after.shape[1] - 1}\n")
        f.write(f"Numeric features: {len(numeric_cols)}\n")
        f.write(f"Categorical features: {len(categorical_cols)}\n")
        f.write(f"Missing values before: {df_before.isnull().sum().sum()}\n")
        f.write(f"Missing values after: {df_after.isnull().sum().sum()}\n")
        f.write(f"Rows before: {df_before.shape[0]}\n")
        f.write(f"Rows after: {df_after.shape[0]}\n")

        f.write("\nSelected features:\n")
        for feat in selected_features:
            f.write(f"- {feat}\n")

        f.write("\nClass distribution:\n")
        for cls, cnt in target_counts.items():
            f.write(f"- {cls}: {cnt}\n")


# --------------------------------------------
# 3. FEATURE SELECTION
# --------------------------------------------

def get_manual_dataset(df):
    required_cols = MANUAL_FEATURES + [TARGET_COL]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Manual mode is missing required columns: {missing_required}")

    df_manual = df[required_cols].copy()
    df_manual = df_manual.dropna(subset=[TARGET_COL]).copy()

    return df_manual, MANUAL_FEATURES


def get_auto_dataset(df):
    df_auto = df.dropna(subset=[TARGET_COL]).copy()
    missing_percent = 100.0 * df_auto.isnull().mean()

    selected_features = []
    for col in df_auto.columns:
        if col == TARGET_COL:
            continue
        if col in UNNECESSARY_COLUMNS:
            continue
        if missing_percent[col] < AUTO_MAX_MISSING_PERCENT:
            selected_features.append(col)

    if not selected_features:
        raise ValueError("Auto mode selected zero features.")

    keep_cols = selected_features + [TARGET_COL]
    df_auto = df_auto[keep_cols].copy()

    return df_auto, selected_features


# --------------------------------------------
# 4. BUILD ALL DATASETS
# --------------------------------------------

def build_manual_imputed(df):
    df_manual, selected_features = get_manual_dataset(df)
    print_preprocessing_diagnostics(df_manual, "manual")
    df_after = impute_dataset(df_manual, TARGET_COL)
    df_after.to_csv(MANUAL_IMPUTED_DIR / "dataset.csv", index=False)

    save_outputs(
        df_before=df_manual,
        df_after=df_after,
        target_col=TARGET_COL,
        selected_features=selected_features,
        out_dir=MANUAL_IMPUTED_DIR,
        mode_name="manual",
        strategy_name="imputed",
    )
    return df_after


def build_manual_dropna(df):
    df_manual, selected_features = get_manual_dataset(df)
    df_after = df_manual.dropna().copy()
    df_after.to_csv(MANUAL_DROPNA_DIR / "dataset.csv", index=False)

    save_outputs(
        df_before=df_manual,
        df_after=df_after,
        target_col=TARGET_COL,
        selected_features=selected_features,
        out_dir=MANUAL_DROPNA_DIR,
        mode_name="manual",
        strategy_name="dropna",
    )
    return df_after


def build_auto_imputed(df):
    df_auto, selected_features = get_auto_dataset(df)
    print_preprocessing_diagnostics(df_auto, "auto")
    df_after = impute_dataset(df_auto, TARGET_COL)
    df_after.to_csv(AUTO_IMPUTED_DIR / "dataset.csv", index=False)

    save_outputs(
        df_before=df_auto,
        df_after=df_after,
        target_col=TARGET_COL,
        selected_features=selected_features,
        out_dir=AUTO_IMPUTED_DIR,
        mode_name="auto",
        strategy_name="imputed",
    )
    return df_after


def build_auto_dropna(df):
    df_auto, selected_features = get_auto_dataset(df)
    df_after = df_auto.dropna().copy()
    df_after.to_csv(AUTO_DROPNA_DIR / "dataset.csv", index=False)

    save_outputs(
        df_before=df_auto,
        df_after=df_after,
        target_col=TARGET_COL,
        selected_features=selected_features,
        out_dir=AUTO_DROPNA_DIR,
        mode_name="auto",
        strategy_name="dropna",
    )
    return df_after


# --------------------------------------------
# 5. MAIN
# --------------------------------------------

def main():
    ensure_output_dirs()
    df = load_input_dataset()

    print("\n==============================")
    print("FINAL PREPROCESSING")
    print("==============================")
    print(f"Input file: {INPUT_DIR / INPUT_FILENAME}")
    print(f"Input shape: {df.shape}")

    manual_imputed_df = build_manual_imputed(df)
    manual_dropna_df = build_manual_dropna(df)
    auto_imputed_df = build_auto_imputed(df)
    auto_dropna_df = build_auto_dropna(df)

    print("\n==============================")
    print("FINAL DATASET SHAPES")
    print("==============================")
    print(f"manual_imputed: {manual_imputed_df.shape}")
    print(f"manual_dropna:  {manual_dropna_df.shape}")
    print(f"auto_imputed:   {auto_imputed_df.shape}")
    print(f"auto_dropna:    {auto_dropna_df.shape}")

    print("\nOutputs saved in:")
    print(f"- {MANUAL_IMPUTED_DIR}")
    print(f"- {MANUAL_DROPNA_DIR}")
    print(f"- {AUTO_IMPUTED_DIR}")
    print(f"- {AUTO_DROPNA_DIR}")


if __name__ == "__main__":
    main()