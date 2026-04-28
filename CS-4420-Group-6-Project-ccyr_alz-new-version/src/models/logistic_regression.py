# ============================================
# LOGISTIC REGRESSION MODEL
# ============================================

from sklearn.linear_model import LogisticRegression


def build_model(params):
    """
    Build logistic regression model from parameter dictionary.
    """
    return LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        solver=params["solver"],
        multi_class=params["multi_class"],
        random_state=params["random_state"],
    )