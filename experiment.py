import numpy as np
import json
from dataclasses import dataclass, field, asdict

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from evaluate import evaluate, positive_scores
from feature_selector import FeatureSelector
from model_factory import create_model
from preprocessor import Preprocessor


# Data class to store information about an experiment's configuration details
# 'Mine' refers to whether to use my implementation as opposed to a custom
@dataclass
class ExperimentConfig:
    experiment_name: str
    scaler_name: str | None = None
    encoder_name: str | None = None
    feature_selector_name: str | None = None
    feature_selector_params: dict = field(default_factory=dict)
    model_name: str = "logistic_regression"
    model_params: dict = field(default_factory=dict)
    mine: bool = False


# Method to run an experiment given a config and some other data
# X, y is passed in to avoid reloading the dataset on every experiment
def run_experiment(
    config: ExperimentConfig,
    X,
    y,
    test_size: float,
    cv_folds: int,
    random_val: int,
):
    # Get the relevant data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_val
    )

    pipeline = Pipeline(
        [
            ("preprocessor", Preprocessor(config.scaler_name, config.encoder_name)),
            (
                "feature_selector",
                FeatureSelector(
                    config.feature_selector_name,
                    config.feature_selector_params,
                    random_val,
                ),
            ),
            (
                "model",
                create_model(
                    config.model_name,
                    config.model_params,
                    random_val,
                    config.mine,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_val)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    model = pipeline.named_steps["model"]
    X_test_processed = pipeline[:-1].transform(X_test)
    y_score = positive_scores(model, X_test_processed)
    test_metrics = evaluate(y_test, y_pred, y_score)

    preprocessor = pipeline.named_steps["preprocessor"]
    selector = pipeline.named_steps["feature_selector"]
    feature_names = preprocessor.get_feature_names_out()
    selected_features = selector.get_selected_feature_names(feature_names)
    feature_ranking = selector.get_feature_ranking(feature_names)

    cv_metrics = {}
    for name, values in cv_result.items():
        if name.startswith("test_"):
            metric_name = name.replace("test_", "")
            cv_metrics[metric_name] = float(np.mean(values))

    results = asdict(config)

    for key, value in cv_metrics.items():
        results[f"cv_{key}"] = value

    for key, value in test_metrics.items():
        results[f"test_{key}"] = value

    results["n_selected_features"] = len(selected_features)
    results["selected_features"] = json.dumps(selected_features)
    results["feature_ranking"] = json.dumps(feature_ranking)

    return results
