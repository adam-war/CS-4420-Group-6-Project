# ============================================
# ADNI DATASET EXPLORATION SCRIPT
# ============================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------
# 0. PATHS
# --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_check"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
CLEAN_DIR = OUTPUT_DIR / "clean_data"

DATASET_FILENAME = "ADNIMERGE_08Jun2025.csv"


def ensure_output_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------

def load_dataset():
    file_path = DATA_DIR / DATASET_FILENAME
    df = pd.read_csv(file_path, low_memory=False)
    return df


# --------------------------------------------
# 2. SAVE BASIC INFORMATION
# --------------------------------------------

def save_basic_info(df):
    print("\n==============================")
    print("DATASET BASIC INFORMATION")
    print("==============================")
    print("Dataset shape (rows, columns):", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    basic_info_df = pd.DataFrame({
        "n_rows": [df.shape[0]],
        "n_columns": [df.shape[1]]
    })
    basic_info_df.to_csv(TABLES_DIR / "dataset_basic_info.csv", index=False)

    pd.DataFrame({"column_name": df.columns}).to_csv(
        TABLES_DIR / "column_names.csv",
        index=False
    )

    pd.DataFrame({
        "column_name": df.columns,
        "dtype": df.dtypes.astype(str).values
    }).to_csv(TABLES_DIR / "column_dtypes.csv", index=False)


# --------------------------------------------
# 3. PATIENT INFORMATION
# --------------------------------------------

def save_patient_info(df):
    print("\n==============================")
    print("PATIENT INFORMATION")
    print("==============================")

    num_patients = df["PTID"].nunique() if "PTID" in df.columns else np.nan

    print("Unique patients:", num_patients)
    print("Total rows (visits):", len(df))

    patient_info_df = pd.DataFrame({
        "unique_patients": [num_patients],
        "total_rows_visits": [len(df)]
    })
    patient_info_df.to_csv(TABLES_DIR / "patient_info.csv", index=False)


# --------------------------------------------
# 4. VISIT DISTRIBUTION
# --------------------------------------------

def save_visit_distribution(df):
    if "VISCODE" not in df.columns:
        print("\nVISCODE column not found. Skipping visit distribution.")
        return

    print("\n==============================")
    print("VISIT DISTRIBUTION")
    print("==============================")

    visit_counts = df["VISCODE"].value_counts(dropna=False)
    print(visit_counts)

    visit_counts_df = visit_counts.reset_index()
    visit_counts_df.columns = ["VISCODE", "count"]
    visit_counts_df.to_csv(TABLES_DIR / "visit_distribution.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(visit_counts_df["VISCODE"].astype(str), visit_counts_df["count"])
    plt.title("Visit Distribution")
    plt.xlabel("VISCODE")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "visit_distribution.png", dpi=300)
    plt.close()


# --------------------------------------------
# 5. BASELINE DATASET
# --------------------------------------------

def get_baseline_dataset(df):
    if "VISCODE" not in df.columns:
        raise ValueError("VISCODE column not found in dataset.")

    df_bl = df[df["VISCODE"] == "bl"].copy()

    print("\n==============================")
    print("BASELINE DATASET")
    print("==============================")
    print("Baseline shape:", df_bl.shape)

    baseline_info_df = pd.DataFrame({
        "baseline_rows": [df_bl.shape[0]],
        "baseline_columns": [df_bl.shape[1]]
    })
    baseline_info_df.to_csv(TABLES_DIR / "baseline_info.csv", index=False)

    return df_bl


# --------------------------------------------
# 6. MISSING VALUES
# --------------------------------------------

def save_missing_values(df_bl):
    print("\n==============================")
    print("MISSING VALUES")
    print("==============================")

    missing_count = df_bl.isnull().sum()
    missing_percent = 100 * missing_count / len(df_bl)

    missing_df = pd.DataFrame({
        "feature": df_bl.columns,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values
    }).sort_values(by="missing_percent", ascending=False)

    print(missing_df.head(20))
    missing_df.to_csv(TABLES_DIR / "missing_values_baseline.csv", index=False)

    top_missing = missing_df.head(20).copy()
    top_missing = top_missing.sort_values(by="missing_percent", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top_missing["feature"], top_missing["missing_percent"])
    plt.title("Top 20 Features with Highest Missing Percentage")
    plt.xlabel("Missing Percentage")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top20_missing_features.png", dpi=300)
    plt.close()


# --------------------------------------------
# 7. FEATURE TYPES
# --------------------------------------------

def save_feature_types(df_bl):
    print("\n==============================")
    print("FEATURE TYPES")
    print("==============================")

    numeric_features = df_bl.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_bl.select_dtypes(exclude=[np.number]).columns.tolist()

    print("\nNumeric features:", len(numeric_features))
    print(numeric_features)

    print("\nCategorical features:", len(categorical_features))
    print(categorical_features)

    pd.DataFrame({"numeric_features": numeric_features}).to_csv(
        TABLES_DIR / "numeric_features.csv",
        index=False
    )

    pd.DataFrame({"categorical_features": categorical_features}).to_csv(
        TABLES_DIR / "categorical_features.csv",
        index=False
    )

    return numeric_features, categorical_features


# --------------------------------------------
# 8. BASIC STATISTICS
# --------------------------------------------

def save_numeric_statistics(df_bl, numeric_features):
    print("\n==============================")
    print("NUMERIC FEATURE STATISTICS")
    print("==============================")

    stats_df = df_bl[numeric_features].describe().T
    print(stats_df)
    stats_df.to_csv(TABLES_DIR / "numeric_feature_statistics.csv")


# --------------------------------------------
# 9. AGE DISTRIBUTION
# --------------------------------------------

def save_age_distribution(df_bl):
    if "AGE" not in df_bl.columns:
        print("\nAGE column not found. Skipping age distribution plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(df_bl["AGE"].dropna(), bins=30)
    plt.title("Age Distribution at Baseline")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "age_distribution.png", dpi=300)
    plt.close()


# --------------------------------------------
# 10. TARGET COMPARISON
# --------------------------------------------

def analyze_target(df_bl, target_col):
    summary = {}

    total_baseline_rows = len(df_bl)
    missing_target = df_bl[target_col].isnull().sum() if target_col in df_bl.columns else np.nan

    if target_col not in df_bl.columns:
        print(f"\nTarget column {target_col} not found.")
        return None

    df_target = df_bl.dropna(subset=[target_col]).copy()

    class_counts = df_target[target_col].value_counts()
    class_percent = 100 * class_counts / len(df_target)

    print(f"\n==============================")
    print(f"TARGET ANALYSIS: {target_col}")
    print("==============================")
    print("Total baseline rows:", total_baseline_rows)
    print("Missing target labels:", missing_target)
    print("Final rows after dropping missing target:", len(df_target))
    print("\nClass counts:")
    print(class_counts)

    summary["target"] = target_col
    summary["baseline_rows"] = total_baseline_rows
    summary["missing_target"] = missing_target
    summary["final_rows"] = len(df_target)
    summary["n_classes"] = class_counts.shape[0]
    summary["classes"] = ", ".join(class_counts.index.astype(str).tolist())

    distribution_df = pd.DataFrame({
        target_col: class_counts.index.astype(str),
        "count": class_counts.values,
        "percentage": class_percent.values
    })

    distribution_df.to_csv(TABLES_DIR / f"{target_col}_distribution.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(distribution_df[target_col], distribution_df["count"])
    plt.title(f"Class Distribution for {target_col} at Baseline")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{target_col}_distribution.png", dpi=300)
    plt.close()

    return df_target, summary


def save_target_comparison(summary_dx, summary_dx_bl):
    rows = []
    if summary_dx is not None:
        rows.append(summary_dx)
    if summary_dx_bl is not None:
        rows.append(summary_dx_bl)

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(TABLES_DIR / "target_comparison.csv", index=False)

    print("\n==============================")
    print("TARGET COMPARISON")
    print("==============================")
    print(comparison_df)


# --------------------------------------------
# 11. SELECTED FEATURE CORRELATION
# --------------------------------------------

def save_selected_feature_correlation(df_bl):
    print("\n==============================")
    print("CORRELATION ANALYSIS")
    print("==============================")

    important_features = [
        "AGE",
        "MMSE",
        "ADAS13",
        "MOCA",
        "FAQ",
        "Hippocampus",
        "Ventricles",
        "WholeBrain"
    ]

    important_features = [f for f in important_features if f in df_bl.columns]

    if len(important_features) < 2:
        print("Not enough selected numeric features for correlation plot.")
        return

    corr = df_bl[important_features].corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(important_features)), important_features, rotation=45, ha="right")
    plt.yticks(range(len(important_features)), important_features)
    plt.title("Selected Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "selected_feature_correlation.png", dpi=300)
    plt.close()

    corr.to_csv(TABLES_DIR / "selected_feature_correlation.csv")


# --------------------------------------------
# 12. FEATURE VS TARGET PLOTS
# --------------------------------------------

def save_boxplots_by_target(df_target, target_col):
    candidate_features = ["MMSE", "ADAS13", "MOCA", "FAQ"]

    for feature in candidate_features:
        if feature not in df_target.columns:
            continue

        plot_df = df_target[[target_col, feature]].dropna()
        if plot_df.empty:
            continue

        classes = plot_df[target_col].astype(str).unique().tolist()
        data_by_class = [
            plot_df.loc[plot_df[target_col].astype(str) == cls, feature].values
            for cls in classes
        ]

        plt.figure(figsize=(7, 4))
        plt.boxplot(data_by_class)
        plt.xticks(range(1, len(classes) + 1), classes)

        plt.title(f"{feature} by {target_col}")
        plt.xlabel(target_col)
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{feature}_vs_{target_col}.png", dpi=300)
        plt.close()


# --------------------------------------------
# 13. CLEAN DATASET EXPORT
# --------------------------------------------

def save_clean_candidate_dataset(df_target, target_col):
    features = [
        "AGE",
        "PTGENDER",
        "PTEDUCAT",
        "APOE4",
        "MMSE",
        "ADAS13",
        "MOCA",
        "FAQ",
        "Hippocampus",
        "Ventricles",
        "WholeBrain",
        target_col
    ]

    features = [f for f in features if f in df_target.columns]

    clean_df = df_target[features].copy()
    clean_df.to_csv(CLEAN_DIR / f"clean_dataset_{target_col}.csv", index=False)

    print(f"\nClean candidate dataset saved: clean_dataset_{target_col}.csv")
    return clean_df


# --------------------------------------------
# 14. MAIN
# --------------------------------------------

def main():
    ensure_output_dirs()

    df = load_dataset()

    save_basic_info(df)
    save_patient_info(df)
    save_visit_distribution(df)

    df_bl = get_baseline_dataset(df)

    if "DX" in df_bl.columns:
        print("\nBaseline DX distribution:")
        print(df_bl["DX"].value_counts(dropna=False))

    if "DX_bl" in df_bl.columns:
        print("\nBaseline DX_bl distribution:")
        print(df_bl["DX_bl"].value_counts(dropna=False))

    save_missing_values(df_bl)
    numeric_features, categorical_features = save_feature_types(df_bl)
    save_numeric_statistics(df_bl, numeric_features)
    save_age_distribution(df_bl)
    save_selected_feature_correlation(df_bl)

    result_dx = analyze_target(df_bl, "DX")
    result_dx_bl = analyze_target(df_bl, "DX_bl")

    df_dx, summary_dx = (result_dx if result_dx is not None else (None, None))
    df_dx_bl, summary_dx_bl = (result_dx_bl if result_dx_bl is not None else (None, None))

    save_target_comparison(summary_dx, summary_dx_bl)

    if df_dx is not None:
        save_boxplots_by_target(df_dx, "DX")
        save_clean_candidate_dataset(df_dx, "DX")

    if df_dx_bl is not None:
        save_boxplots_by_target(df_dx_bl, "DX_bl")
        save_clean_candidate_dataset(df_dx_bl, "DX_bl")

    print("\nDataset check completed.")
    print(f"Tables saved in: {TABLES_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Clean datasets saved in: {CLEAN_DIR}")


if __name__ == "__main__":
    main()