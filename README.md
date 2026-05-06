# Breast Cancer Classification Experiments

This project compares several classical machine learning models on the Breast Cancer Wisconsin Original dataset. The goal is to evaluate how different preprocessing choices, feature selection methods, and model families affect binary classification performance for benign vs. malignant tumor prediction.

The project uses a reproducible experiment pipeline built with Python and scikit-learn. It also includes custom implementations of several models to compare against scikit-learn baselines.

## Dataset

This project uses the Breast Cancer Wisconsin Original dataset from the UCI Machine Learning Repository.

The dataset contains:

- 699 samples
- 9 predictive features
- A binary target class: benign or malignant
- Missing values represented with `?`, primarily in the `bare_nuclei` feature

The data loading script converts missing values to null values, coerces feature columns to numeric values, and maps the class labels into binary values.

## Models Compared

The experiments compare the following model families:

- Logistic Regression
- Ridge Logistic Regression
- Gaussian Naive Bayes
- k-Nearest Neighbors
- Decision Tree
- Random Forest

The project also includes custom implementations of:

- Logistic Regression
- Ridge Logistic Regression
- Gaussian Naive Bayes
- k-Nearest Neighbors

These custom models are implemented with `fit`, `predict`, and `predict_proba` methods so they can be used inside the same experiment workflow as the scikit-learn models.

## Experiment Pipeline

Each experiment follows the same general pipeline:

1. Load the dataset
2. Split the data into training and testing sets
3. Apply preprocessing
4. Optionally apply feature selection
5. Train the model
6. Evaluate using cross-validation and test-set metrics
7. Save result tables, summaries, and plots

The main experiment runner builds a list of model configurations and runs each one through a shared scikit-learn pipeline.

## Preprocessing

The preprocessing step supports:

- Median imputation for numeric features
- Most-frequent imputation when using one-hot encoding
- Optional one-hot encoding
- Optional scaling with:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler

Scaling is especially important for logistic regression and k-nearest neighbors.

## Feature Selection

The project tests several feature selection strategies:

- No feature selection
- Mutual information top-k selection
- Random forest importance top-k selection
- Recursive feature elimination with logistic regression

Selected feature names and feature rankings are saved with the experiment results.

## Evaluation Metrics

Each model is evaluated with 5-fold stratified cross-validation and a held-out test set.

The recorded metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix values: true positives, false positives, true negatives, false negatives

The results are saved under the `results/` directory.

## Example Results

In the final experiment set, ridge logistic regression produced the strongest overall balance between ROC-AUC and accuracy. Gaussian Naive Bayes performed especially well when optimizing for recall.

The custom implementations produced results that were broadly similar to the corresponding scikit-learn models, which helped validate the correctness of the custom implementations.

## Project Structure

```text
.
├── breast-cancer-wisconsin.data
├── main.py
├── load_data.py
├── experiment.py
├── preprocessor.py
├── feature_selector.py
├── model_factory.py
├── evaluate.py
├── summarize_results.py
├── requirements.txt
└── results/
    ├── experiment_results.csv
    ├── tables/
    ├── plots/
    └── summaries/
