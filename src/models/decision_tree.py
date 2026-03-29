# ============================================
# DECISION TREE MODEL
# ============================================

from sklearn.tree import DecisionTreeClassifier


def build_model(params):
    """
    Build Decision Tree model from parameter dictionary.
    """
    return DecisionTreeClassifier(
        criterion=params["criterion"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )