# ============================================
# SGD CLASSIFIER MODEL
# ============================================

from sklearn.linear_model import SGDClassifier


def build_model(params):
    """
    Build SGD Classifier model from parameter dictionary.
    """
    return SGDClassifier(
        loss=params["loss"],
        alpha=params["alpha"],
        max_iter=params["max_iter"],
        tol=params["tol"],
        random_state=params["random_state"],
    )