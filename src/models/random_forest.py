# ============================================
# RANDOM FOREST MODEL
# ============================================

from sklearn.ensemble import RandomForestClassifier


def build_model(params):
    """
    Build Random Forest model from parameter dictionary.
    """
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        criterion=params["criterion"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
    )