# ============================================
# ADALINE MODEL
# ============================================

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class AdalineClassifier(BaseEstimator, ClassifierMixin):
    """
    Multiclass Adaline classifier using one-vs-rest strategy.

    Parameters
    ----------
    eta : float
        Learning rate.
    n_iter : int
        Number of epochs.
    random_state : int
        Random seed.
    """

    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def _fit_binary(self, X, y_binary):
        """
        Train one binary Adaline classifier.
        Positive class is encoded as +1, negative class as -1.
        """
        rgen = np.random.RandomState(self.random_state)
        w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        b = np.float64(0.0)

        for _ in range(self.n_iter):
            net_input = np.dot(X, w) + b
            output = net_input
            errors = y_binary - output

            w += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            b += self.eta * 2.0 * errors.mean()

        return w, b

    def fit(self, X, y):
        """
        Fit multiclass Adaline using one-vs-rest.
        """
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)

        self.classes_ = self.label_encoder_.classes_
        self.weights_ = []
        self.biases_ = []

        for class_idx in range(len(self.classes_)):
            y_binary = np.where(y_encoded == class_idx, 1.0, -1.0)
            w, b = self._fit_binary(X, y_binary)
            self.weights_.append(w)
            self.biases_.append(b)

        self.weights_ = np.array(self.weights_)
        self.biases_ = np.array(self.biases_)

        return self

    def decision_function(self, X):
        """
        Compute scores for each class.
        """
        return np.dot(X, self.weights_.T) + self.biases_

    def predict(self, X):
        """
        Predict class labels.
        """
        scores = self.decision_function(X)
        pred_idx = np.argmax(scores, axis=1)
        return self.label_encoder_.inverse_transform(pred_idx)

    def predict_proba(self, X):
        """
        Approximate probabilities using softmax over class scores.
        """
        scores = self.decision_function(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs


def build_model(params):
    """
    Build Adaline model from parameter dictionary.
    """
    return AdalineClassifier(
        eta=params["eta"],
        n_iter=params["n_iter"],
        random_state=params["random_state"],
    )