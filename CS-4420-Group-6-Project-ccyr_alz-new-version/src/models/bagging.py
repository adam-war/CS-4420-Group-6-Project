# ============================================
# BAGGING MODEL
# ============================================
# BAGGING MODEL
# ============================================

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def build_model(params):
    """
    Build Bagging model using Decision Trees as base estimator.
    Compatible with older scikit-learn versions.
    """
    base_estimator = DecisionTreeClassifier(random_state=params["random_state"])

    return BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=params["n_estimators"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
    )