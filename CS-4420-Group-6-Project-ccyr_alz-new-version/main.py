# ============================================
# MAIN
# ============================================

from src.evaluation.model_runner import run_all_models


def main():
    overall_summary_df, run_dir, run_name = run_all_models()

    print("\n==============================")
    print("RUN COMPLETED")
    print("==============================")
    print(f"Run name: {run_name}")
    print(f"Results directory: {run_dir}")


if __name__ == "__main__":
    main()