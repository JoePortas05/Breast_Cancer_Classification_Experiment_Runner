import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression


class FeatureSelector(BaseEstimator):
    def __init__(
        self,
        selector_name: str | None = None,
        selector_params: dict | None = None,
        random_state: int = 1,
    ) -> None:
        self.selector_name = selector_name
        self.selector_params = selector_params
        self.random_state = random_state

    def fit(self, X, y):
        n_features = X.shape[1]

        if self.selector_name in (None, "none"):
            self.support_mask_ = np.ones(n_features, dtype=bool)
            self.feature_scores_ = np.ones(n_features, dtype=float)
            return self

        if self.selector_params is None:
            params = {}
        else:
            params = self.selector_params

        k = min(int(params.get("k", n_features)), n_features)

        if self.selector_name == "mutual_info_top_k":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selector.fit(X, y)
            self.support_mask_ = selector.get_support()
            self.feature_scores_ = np.nan_to_num(selector.scores_, nan=0.0)
            return self

        if self.selector_name == "rf_top_k":
            model = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state
            )
            model.fit(X, y)
            self.feature_scores_ = model.feature_importances_
            ranked_indices = np.argsort(self.feature_scores_)[::-1]
            top_indices = ranked_indices[:k]
            self.support_mask_ = np.zeros(n_features, dtype=bool)
            self.support_mask_[top_indices] = True
            return self

        if self.selector_name == "rfe":
            estimator = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
            selector = RFE(estimator=estimator, n_features_to_select=k)
            selector.fit(X, y)
            self.support_mask_ = selector.get_support()
            self.feature_scores_ = 1.0 / selector.ranking_
            return self

        raise ValueError(f"Unknown feature selector: {self.selector_name}")

    def transform(self, X):
        return X[:, self.support_mask_]

    def get_selected_feature_names(self, input_feature_names):
        names = np.asarray(input_feature_names, dtype=object)
        return names[self.support_mask_].tolist()

    def get_feature_ranking(self, input_feature_names):
        names = np.asarray(input_feature_names, dtype=object)

        names_list = names.tolist()
        fs_list = self.feature_scores_.tolist()
        pairs = list(zip(names_list, fs_list))
        pairs.sort(key=lambda item: item[1], reverse=True)

        ranking = {}
        for name, score in pairs:
            ranking[name] = float(score)

        return ranking
