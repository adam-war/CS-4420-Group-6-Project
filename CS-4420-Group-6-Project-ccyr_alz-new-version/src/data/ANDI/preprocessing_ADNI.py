# ============================================
# FINAL PREPROCESSING SCRIPT
# ============================================

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer



# --------------------------------------------
# PATHS
# --------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path('.')


AUTO_MAX_MISSING_PERCENT = 10.0


# --------------------------------------------
# CONFIGURATION
# --------------------------------------------


@dataclass
class Dataset:
  dataset_folder_name: str
  input_filename: str
  target_column: str
  manual_features: list[str]
  unnecessary_columns: list[str]

  def __post_init__(self):
    self.input_dir = PROJECT_ROOT / self.dataset_folder_name / "outputs" / "dataset_check" / "clean_data"
    self.output_dir = PROJECT_ROOT / self.dataset_folder_name / "outputs" / "preprocessing"
    self.manual_imputed_dir = self.output_dir / "manual_imputed"
    self.manual_dropna_dir = self.output_dir / "manual_dropna"
    self.auto_imputed_dir = self.output_dir / "auto_imputed"
    self.auto_dropna_dir = self.output_dir / "auto_dropna"

  # --------------------------------------------
  # HELPERS
  # --------------------------------------------

  def ensure_output_dirs(self):
      self.input_dir.mkdir(parents=True, exist_ok=True)
      self.output_dir.mkdir(parents=True, exist_ok=True)
      self.manual_imputed_dir.mkdir(parents=True, exist_ok=True)
      self.manual_dropna_dir.mkdir(parents=True, exist_ok=True)
      self.auto_imputed_dir.mkdir(parents=True, exist_ok=True)
      self.auto_dropna_dir.mkdir(parents=True, exist_ok=True)


  def load_input_dataset(self):
      file_path =  self.input_dir / self.input_filename
      if not file_path.exists():
          raise FileNotFoundError(
              f"Input dataset not found: {file_path}\n"
              f"Make sure dataset_check already created {self.input_filename}."
          )
      self.df = pd.read_csv(file_path)



  def build_manual_imputed(self):
      df_manual = self.get_manual_dataset()
      self.print_preprocessing_diagnostics(df_manual, "manual")
      df_after = self.impute_dataset(df_manual)
      df_after.to_csv(self.manual_imputed_dir / "dataset.csv", index=False)

      self.save_outputs(
          df_before=df_manual,
          df_after=df_after,
          selected_features=self.manual_features,
          out_dir=self.manual_imputed_dir,
          mode_name="manual",
          strategy_name="imputed",
      )
      return df_after


  def build_manual_dropna(self):
      df_manual = self.get_manual_dataset()
      df_after = df_manual.dropna().copy()
      df_after.to_csv(self.manual_dropna_dir / "dataset.csv", index=False)

      self.save_outputs(
          df_before=df_manual,
          df_after=df_after,
          selected_features=self.manual_features,
          out_dir=self.manual_dropna_dir,
          mode_name="manual",
          strategy_name="dropna",
      )
      return df_after


  def build_auto_imputed(self):
      df_auto, selected_features = self.get_auto_dataset()
      self.print_preprocessing_diagnostics(df_auto, "auto")
      df_after = self.impute_dataset(df_auto)
      df_after.to_csv(self.auto_imputed_dir / "dataset.csv", index=False)

      self.save_outputs(
          df_before=df_auto,
          df_after=df_after,
          selected_features=selected_features,
          out_dir=self.auto_imputed_dir,
          mode_name="auto",
          strategy_name="imputed",
      )
      return df_after


  def build_auto_dropna(self):
      df_auto, selected_features = self.get_auto_dataset()
      df_after = df_auto.dropna().copy()
      df_after.to_csv(self.auto_imputed_dir / "dataset.csv", index=False)

      self.save_outputs(
          df_before=df_auto,
          df_after=df_after,
          selected_features=selected_features,
          out_dir=self.auto_dropna_dir,
          mode_name="auto",
          strategy_name="dropna",
      )
      return df_after

  def compute_missing_summary(self, df):
      missing_count = df.isnull().sum()
      missing_percent = 100.0 * missing_count / len(df)

      summary_df = pd.DataFrame({
          "feature": df.columns,
          "missing_count": missing_count.values,
          "missing_percent": missing_percent.values,
      }).sort_values(by=["missing_percent", "feature"], ascending=[False, True])

      return summary_df


  def split_feature_types(self, df):
      feature_df = df.drop(columns=[self.target_column], errors="ignore")
      numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
      categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
      return numeric_cols, categorical_cols


  def impute_dataset(self, df):
      df_out = df.copy()

      feature_cols = [col for col in df_out.columns if col != self.target_column]

      numeric_cols = df_out[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
      categorical_cols = df_out[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

      if numeric_cols:
          num_imputer = SimpleImputer(strategy="mean")
          df_out[numeric_cols] = num_imputer.fit_transform(df_out[numeric_cols])

      if categorical_cols:
          cat_imputer = SimpleImputer(strategy="most_frequent")
          df_out[categorical_cols] = cat_imputer.fit_transform(df_out[categorical_cols])

      return df_out


  def print_preprocessing_diagnostics(self, df_before, mode_name):
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


  def save_outputs(self, df_before, df_after, selected_features, out_dir, mode_name, strategy_name):
      missing_before_df = self.compute_missing_summary(df_before)
      missing_after_df = self.compute_missing_summary(df_after)

      missing_before_df.to_csv(out_dir / "missing_before.csv", index=False)
      missing_after_df.to_csv(out_dir / "missing_after.csv", index=False)

      feature_df = pd.DataFrame({"feature": selected_features})
      feature_df.to_csv(out_dir / "selected_features.csv", index=False)

      preview_df = df_after.head(10)
      preview_df.to_csv(out_dir / "dataset_preview.csv", index=False)

      target_counts = df_after[self.target_column].value_counts(dropna=False)
      target_percent = 100.0 * target_counts / len(df_after)

      target_dist_df = pd.DataFrame({
          self.target_column: target_counts.index.astype(str),
          "count": target_counts.values,
          "percentage": target_percent.values,
      })
      target_dist_df.to_csv(out_dir / "target_distribution.csv", index=False)

      numeric_cols, categorical_cols = self.split_feature_types(df_after)

      summary = {
          "mode": mode_name,
          "strategy": strategy_name,
          "target": self.target_column,
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
          f.write(f"Target: {self.target_column}\n")
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
  # FEATURE SELECTION
  # --------------------------------------------

  def get_manual_dataset(self):
      required_cols = self.manual_features + [self.target_column]
      missing_required = [col for col in required_cols if col not in self.df.columns]
      if missing_required:
          raise ValueError(f"Manual mode is missing required columns: {missing_required}")
      df_manual = self.df[required_cols].copy()
      df_manual = df_manual.dropna(subset=[self.target_column]).copy()

      return df_manual


  def get_auto_dataset(self):
      
      df_auto = self.df.dropna(subset=[self.target_column]).copy()
      missing_percent = 100.0 * df_auto.isnull().mean()

      selected_features = []
      for col in df_auto.columns:
          if col == self.target_column:
              continue
          if col in self.unnecessary_columns:
              continue
          if missing_percent[col] < AUTO_MAX_MISSING_PERCENT:
              selected_features.append(col)

      if not selected_features:
          raise ValueError("Auto mode selected zero features.")

      keep_cols = selected_features + [self.target_column]
      df_auto = df_auto[keep_cols].copy()

      return df_auto, selected_features


  # --------------------------------------------
  # BUILD ALL DATASETS
  # --------------------------------------------
  def run_pipeline(self):
      self.ensure_output_dirs()
      self.load_input_dataset()

      print("\n==============================")
      print("FINAL PREPROCESSING")
      print("==============================")
      print(f"Input file: {self.input_dir / self.input_filename}")
      print(f"Input shape: {self.df.shape}")

      manual_imputed_df = self.build_manual_imputed()
      manual_dropna_df = self.build_manual_dropna()
      auto_imputed_df = self.build_auto_imputed()
      auto_dropna_df = self.build_auto_dropna()

      print("\n==============================")
      print("FINAL DATASET SHAPES")
      print("==============================")
      print(f"manual_imputed: {manual_imputed_df.shape}")
      print(f"manual_dropna:  {manual_dropna_df.shape}")
      print(f"auto_imputed:   {auto_imputed_df.shape}")
      print(f"auto_dropna:    {auto_dropna_df.shape}")

      print("\nOutputs saved in:")
      print(f"- {self.manual_imputed_dir}")
      print(f"- {self.manual_dropna_dir}")
      print(f"- {self.auto_imputed_dir}")
      print(f"- {self.auto_dropna_dir}")



# --------------------------------------------
# MAIN
# --------------------------------------------

def main():
  target_column = "DX"

  input_filename = "clean_dataset_DX.csv"
  manual_features = [
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

  unnecessary_columns = [
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
  ]
  dataset_folder_name = "test"

  alz1 = Dataset(target_column=target_column, 
                   input_filename=input_filename, 
                   manual_features=manual_features, 
                   unnecessary_columns=unnecessary_columns, dataset_folder_name=dataset_folder_name)
  alz1.run_pipeline()


if __name__ == "__main__":
    main()