import random
import numpy as np
import pandas as pd
import os

from experiment import run_experiment, ExperimentConfig
from summarize_results import summarize_results
from load_data import load_dataset

# Constants
RANDOM_STATE = 1
TEST_SIZE = 0.2
CV_FOLDS = 5
TOP_K = 5


def build_experiments() -> list[ExperimentConfig]:
    experiments = []

    linear_scalers = [None, "standard", "minmax", "robust"]
    selector_options = [
        (None, None),
        ("mutual_info_top_k", {"k": TOP_K}),
        ("rf_top_k", {"k": TOP_K}),
        ("rfe", {"k": TOP_K}),
    ]

    for scaler_name in linear_scalers:

        if scaler_name is not None:
            name = scaler_name
        else:
            name = "none"

        experiments.append(
            ExperimentConfig(
                experiment_name=f"logreg_scaler_{name}",
                model_name="logistic_regression",
                scaler_name=scaler_name,
            )
        )

    for scaler_name in linear_scalers:
        for c_value in [0.1, 1.0, 10.0]:

            if scaler_name is not None:
                name = scaler_name
            else:
                name = "none"

            experiments.append(
                ExperimentConfig(
                    experiment_name=f"ridge_scaler_{name}_c_{c_value}",
                    model_name="ridge_logistic_regression",
                    model_params={"C": c_value},
                    scaler_name=scaler_name,
                )
            )

    for scaler_name in linear_scalers:
        if scaler_name is not None:
            name = scaler_name
        else:
            name = "none"

        experiments.append(
            ExperimentConfig(
                experiment_name=f"gnb_scaler_{name}",
                model_name="gaussian_nb",
                scaler_name=scaler_name,
            )
        )

    for scaler_name in linear_scalers:
        for neighbors in [3, 5, 7, 9]:
            for weights in ["uniform", "distance"]:

                if scaler_name is not None:
                    name = scaler_name
                else:
                    name = "none"

                experiments.append(
                    ExperimentConfig(
                        experiment_name=f"knn_scaler_{name}_k_{neighbors}_{weights}",
                        model_name="knn",
                        model_params={"n_neighbors": neighbors, "weights": weights},
                        scaler_name=scaler_name,
                    )
                )

    for max_depth in [None, 3, 5, 7]:
        for min_samples_leaf in [1, 3]:
            for selector_name, selector_params in selector_options:

                if max_depth is not None:
                    depth = max_depth
                else:
                    depth = "none"

                if selector_name is not None:
                    name = selector_name
                else:
                    name = "none"

                experiments.append(
                    ExperimentConfig(
                        experiment_name=f"tree_depth_{depth}_leaf_{min_samples_leaf}_sel_{name}",
                        model_name="decision_tree",
                        model_params={
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                        },
                        feature_selector_name=selector_name,
                        feature_selector_params=selector_params,
                    )
                )

    for n_estimators in [100, 200]:
        for max_depth in [None, 3, 5]:
            for selector_name, selector_params in selector_options:

                if max_depth is not None:
                    depth = max_depth
                else:
                    depth = "none"

                if selector_name is not None:
                    name = selector_name
                else:
                    name = "none"

                experiments.append(
                    ExperimentConfig(
                        experiment_name=f"rf_trees_{n_estimators}_depth_{depth}_sel_{name}",
                        model_name="random_forest",
                        model_params={
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                        },
                        feature_selector_name=selector_name,
                        feature_selector_params=selector_params,
                    )
                )

    for model_name in [
        "logistic_regression",
        "ridge_logistic_regression",
        "gaussian_nb",
        "knn",
    ]:
        model_params = {}
        if model_name == "ridge_logistic_regression":
            model_params = {"C": 1.0}
        elif model_name == "knn":
            model_params = {"n_neighbors": 5, "weights": "uniform"}

        experiments.append(
            ExperimentConfig(
                experiment_name=f"{model_name}_onehot",
                model_name=model_name,
                model_params=model_params,
                encoder_name="onehot",
            )
        )

    for selector_name, selector_params in selector_options[1:]:
        experiments.append(
            ExperimentConfig(
                experiment_name=f"ridge_standard_sel_{selector_name}",
                model_name="ridge_logistic_regression",
                model_params={"C": 1.0},
                scaler_name="standard",
                feature_selector_name=selector_name,
                feature_selector_params=selector_params,
            )
        )

    experiments.extend(
        [
            ExperimentConfig(
                experiment_name="custom_logreg_standard",
                model_name="logistic_regression",
                scaler_name="standard",
                mine=True,
            ),
            ExperimentConfig(
                experiment_name="custom_ridge_robust_c_0.1",
                model_name="ridge_logistic_regression",
                model_params={"C": 0.1},
                scaler_name="robust",
                mine=True,
            ),
            ExperimentConfig(
                experiment_name="custom_gnb_none", model_name="gaussian_nb", mine=True
            ),
            ExperimentConfig(
                experiment_name="custom_knn_standard_k_5_distance",
                model_name="knn",
                model_params={"n_neighbors": 5, "weights": "distance"},
                scaler_name="standard",
                mine=True,
            ),
        ]
    )

    return experiments


def main() -> None:
    X, y = load_dataset()
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    results = []

    experiments = build_experiments()
    total = len(experiments)

    for index, config in enumerate(experiments, start=1):
        result = run_experiment(config, X, y, TEST_SIZE, CV_FOLDS, RANDOM_STATE)
        results.append(result)
        print(f"[{index}/{total}] Ran: {config.experiment_name}")

    os.makedirs("results/", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/experiment_results.csv", index=False)
    summarize_results(df)

    print("Done running experiments")


if __name__ == "__main__":
    main()
