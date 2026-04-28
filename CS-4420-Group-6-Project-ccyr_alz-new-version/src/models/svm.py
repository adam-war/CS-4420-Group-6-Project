# ============================================
# SVM MODELS
# ============================================

from sklearn.svm import SVC


def build_linear_svm(params):
    """
    Build linear SVM from parameter dictionary.
    """
    return SVC(
        C=params["C"],
        kernel=params["kernel"],
        probability=params["probability"],
        random_state=params["random_state"],
    )


def build_rbf_svm(params):
    """
    Build RBF SVM from parameter dictionary.
    """
    return SVC(
        C=params["C"],
        kernel=params["kernel"],
        gamma=params["gamma"],
        probability=params["probability"],
        random_state=params["random_state"],
    )