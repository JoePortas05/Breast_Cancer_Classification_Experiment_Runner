import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


class Preprocessor:
    def __init__(
        self, scaler_name: str | None = None, encoder_name: str | None = None
    ) -> None:
        self.scaler_name = scaler_name
        self.encoder_name = encoder_name

    def fit(self, X, y=None):
        feature_names = np.array(X.columns, dtype=object)

        if self.encoder_name == "onehot":
            strat = "most_frequent"
        else:
            strat = "median"

        steps = [("imputer", SimpleImputer(strategy=strat))]

        if self.encoder_name == "onehot":
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            steps.append(("encoder", onehot))

        scaler = None

        if self.scaler_name == "standard":
            scaler = StandardScaler()
        elif self.scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif self.scaler_name == "robust":
            scaler = RobustScaler()

        if scaler is not None:
            steps.append(("scaler", scaler))

        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X, y)

        if self.encoder_name == "onehot":
            encoder = self.pipeline_.named_steps["encoder"]
            self.output_feature_names_ = np.array(
                encoder.get_feature_names_out(feature_names), dtype=object
            )
        else:
            self.output_feature_names_ = feature_names

        return self

    def transform(self, X):
        Xt = self.pipeline_.transform(X)
        return np.asarray(Xt, dtype=float)

    def get_feature_names_out(self):
        return self.output_feature_names_
