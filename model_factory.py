import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class CustomLogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 2000,
        C: float = 1.0,
        tol: float = 1e-6,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.C = C
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features, dtype=float)
        self.bias = 0.0

        if self.C is None or self.C <= 0:
            reg = 0.0
        else:
            reg = 1.0 / self.C

        prev = 999999999

        for _ in range(self.max_iter):
            scores = X @ self.weights + self.bias
            probs = sigmoid(scores)
            errors = probs - y

            X_e = X.T @ errors
            reg_per = reg / num_samples

            grad_weights = X_e / num_samples + reg_per * self.weights
            grad_bias = float(np.mean(errors))

            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

            probs_log = np.log(probs + 1e-12)
            adj_log = np.log(1.0 - probs + 1e-12)
            avg = np.mean(y * probs_log + (1.0 - y) * adj_log)
            loss = -avg

            weightsss = np.sum(self.weights**2)
            loss += 0.5 * reg * weightsss / num_samples

            loss_dif = prev - loss
            if abs(loss_dif) < self.tol:
                break
            prev = loss

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        X_w = X @ self.weights
        prob = sigmoid(X_w + self.bias)

        prob_adj = 1.0 - prob
        return np.column_stack([prob_adj, prob])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class CustomGaussianNB(ClassifierMixin, BaseEstimator):
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.unique(y)
        self.prior_classes = []
        self.theta = []
        self.var = []

        fvar = np.var(X, axis=0)
        max_fvar = fvar.max()

        for i_class in self.classes_:
            same = y == i_class
            X_class = X[same]

            theta = np.mean(X_class, axis=0)
            self.theta.append(theta)

            var = np.var(X_class, axis=0) + self.var_smoothing * max_fvar
            self.var.append(var)

            prior_class = X_class.shape[0] / X.shape[0]
            self.prior_classes.append(prior_class)

        self.theta = np.asarray(self.theta, dtype=float)
        self.var = np.asarray(self.var, dtype=float)
        self.prior_classes = np.asarray(self.prior_classes, dtype=float)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        all_scores = []

        for idx, _ in enumerate(self.classes_):
            theta = self.theta[idx]
            var = self.var[idx]

            prior_class = self.prior_classes[idx]
            prior_log = np.log(prior_class)

            log = np.log(2.0 * np.pi * var)
            sum = np.sum(log)
            log_likelihood = -0.5 * sum

            sqr = (X - theta) ** 2
            sum = np.sum(sqr / var, axis=1)
            log_likelihood -= 0.5 * sum

            all_scores.append(prior_log + log_likelihood)

        scores = np.column_stack(all_scores)
        max_score = scores.max(axis=1, keepdims=True)
        scores = scores - max_score
        probs = np.exp(scores)
        total_probs = probs.sum(axis=1, keepdims=True)
        return probs / total_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        amax = np.argmax(probs, axis=1)
        return self.classes_[amax]


class CustomKnnClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        probs = np.zeros((X.shape[0], 2), dtype=float)

        for idx, sample in enumerate(X):
            diff = self.X_train - sample
            diff_sqr = diff**2
            train_sum = np.sum(diff_sqr, axis=1)
            distances = np.sqrt(train_sum)

            sorted = np.argsort(distances)
            neighbors = sorted[: self.n_neighbors]
            neighbor_labels = self.y_train[neighbors]
            neighbor_distances = distances[neighbors]

            if self.weights == "distance":
                max = np.maximum(neighbor_distances, 1e-12)
                neighbor_weights = 1.0 / max
            else:
                neighbor_weights = np.ones_like(neighbor_distances)

            sum_at_true = neighbor_weights[neighbor_labels == 1].sum()
            sum_all = neighbor_weights.sum()
            prob = sum_at_true / sum_all

            probs[idx, 1] = prob
            probs[idx, 0] = 1.0 - prob

        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def create_model(
    model_name: str, model_params: dict, random_state: int, mine: bool = False
):
    if model_params is not None:
        params = model_params
    else:
        params = {}

    if model_name == "logistic_regression":
        defaults = {"C": 1e6, "solver": "liblinear", "max_iter": 2000}
        defaults.update(params)

        if mine:
            customs = {
                "learning_rate": 0.05,
                "max_iter": defaults["max_iter"],
                "C": defaults["C"],
                "tol": 1e-6,
            }
            return CustomLogisticRegression(**customs)
        return LogisticRegression(**defaults)

    if model_name == "ridge_logistic_regression":
        defaults = {"C": 1.0, "solver": "liblinear", "max_iter": 2000}
        defaults.update(params)

        if mine:
            customs = {
                "learning_rate": 0.05,
                "max_iter": defaults["max_iter"],
                "C": defaults["C"],
                "tol": 1e-6,
            }
            return CustomLogisticRegression(**customs)
        return LogisticRegression(**defaults)

    if model_name == "gaussian_nb":
        if mine:
            return CustomGaussianNB(**params)
        return GaussianNB(**params)

    if model_name == "knn":
        defaults = {"n_neighbors": 5, "weights": "uniform"}
        defaults.update(params)

        if mine:
            return CustomKnnClassifier(**defaults)
        return KNeighborsClassifier(**defaults)

    if model_name == "decision_tree":
        defaults = {"random_state": random_state}
        defaults.update(params)

        return DecisionTreeClassifier(**defaults)

    if model_name == "random_forest":
        defaults = {"n_estimators": 200, "random_state": random_state}
        defaults.update(params)

        return RandomForestClassifier(**defaults)

    raise ValueError(f"Unknown model name ({model_name})")
