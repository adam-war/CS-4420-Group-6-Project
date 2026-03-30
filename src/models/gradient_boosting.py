# ============================================
# GRADIENT BOOSTING MODEL
# ============================================

from sklearn.ensemble import GradientBoostingClassifier

def build_model(params):
    """
    Build Gradient Boosting model from parameter dictionary.
    """
    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
        loss='log_loss'
    )