# ============================================
# KNN MODEL
# ============================================

from sklearn.neighbors import KNeighborsClassifier


def build_model(params):
    """
    Build KNN model from parameter dictionary.
    """
    return KNeighborsClassifier(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        metric=params["metric"],
        p=params["p"],
    )