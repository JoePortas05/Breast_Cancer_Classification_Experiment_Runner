import os
import matplotlib.pyplot as plt


# Method to summarize the results of running experiments
def summarize_results(df):

    # Set and make relevant filepaths
    tables = "results/tables"
    plots = "results/plots"
    summaries = "results/summaries"
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    os.makedirs(summaries, exist_ok=True)

    # Make a CSV for each metric test
    for metric in ["test_roc_auc", "test_recall", "test_precision", "test_accuracy"]:
        path = os.path.join(tables, f"{metric}.csv")
        cols = [
            "experiment_name",
            "model_name",
            "mine",
            "scaler_name",
            "encoder_name",
            "feature_selector_name",
            "n_selected_features",
            metric,
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "cv_accuracy",
            "cv_precision",
            "cv_recall",
            "cv_f1",
            "cv_roc_auc",
        ]
        df.sort_values(metric, ascending=False)[cols].head(10).to_csv(path, index=False)

    # Get data by for the roc_auc value tests
    best_auc = (
        df.sort_values("test_roc_auc", ascending=False)
        .groupby(["mine", "model_name"], as_index=False)
        .first()
    )

    # Create labels for the figure
    auc_labels = best_auc["mine"].astype(str) + "_" + best_auc["model_name"]

    # Draw the plot for best roc_auc
    plt.figure(figsize=(10, 5))
    plt.bar(auc_labels, best_auc["test_roc_auc"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Test ROC-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(plots, "best_roc_auc_by_model.png"), dpi=200)
    plt.close()

    # Get data by for the recall value tests
    best_recall = (
        df.sort_values("test_recall", ascending=False)
        .groupby(["mine", "model_name"], as_index=False)
        .first()
    )

    # Create labels for the figure
    recall_labels = best_recall["mine"].astype(str) + "_" + best_recall["model_name"]

    # Draw the plot for best recall
    plt.figure(figsize=(10, 5))
    plt.bar(recall_labels, best_recall["test_recall"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Test Recall")
    plt.tight_layout()
    plt.savefig(os.path.join(plots, "best_recall_by_model.png"), dpi=200)
    plt.close()

    # Get best of multiple metric values
    top_test_roc_auc = df.sort_values("test_roc_auc", ascending=False).iloc[0]
    top_test_recall = df.sort_values("test_recall", ascending=False).iloc[0]
    top_test_accuracy = df.sort_values("test_accuracy", ascending=False).iloc[0]
    top_test_precision = df.sort_values("test_precision", ascending=False).iloc[0]
    top_test_f1 = df.sort_values("test_f1", ascending=False).iloc[0]

    top_cv_roc_auc = df.sort_values("cv_roc_auc", ascending=False).iloc[0]
    top_cv_recall = df.sort_values("cv_recall", ascending=False).iloc[0]
    top_cv_accuracy = df.sort_values("cv_accuracy", ascending=False).iloc[0]
    top_cv_precision = df.sort_values("cv_precision", ascending=False).iloc[0]
    top_cv_f1 = df.sort_values("cv_f1", ascending=False).iloc[0]

    top_tp = df.sort_values("test_tp", ascending=False).iloc[0]
    top_tn = df.sort_values("test_tn", ascending=False).iloc[0]
    top_fp = df.sort_values("test_fp", ascending=True).iloc[0]
    top_fn = df.sort_values("test_fn", ascending=True).iloc[0]

    # Write up best values in a short summary text file
    summary_txt = (
        f"TEST:\n"
        f"Best by Test ROC-AUC: {top_test_roc_auc['experiment_name']} ({top_test_roc_auc['test_roc_auc']:.4f})\n"
        f"Best by Test Recall: {top_test_recall['experiment_name']} ({top_test_recall['test_recall']:.4f})\n"
        f"Best by Test Accuracy: {top_test_accuracy['experiment_name']} ({top_test_accuracy['test_accuracy']:.4f})\n"
        f"Best by Test Precision: {top_test_precision['experiment_name']} ({top_test_precision['test_precision']:.4f})\n"
        f"Best by Test F1: {top_test_f1['experiment_name']} ({top_test_f1['test_f1']:.4f})\n\n"
        f"CROSS VALIDATION:\n"
        f"Best by Cross Validation ROC-AUC: {top_cv_roc_auc['experiment_name']} ({top_cv_roc_auc['cv_roc_auc']:.4f})\n"
        f"Best by Cross Validation Recall: {top_cv_recall['experiment_name']} ({top_cv_recall['cv_recall']:.4f})\n"
        f"Best by Cross Validation Accuracy: {top_cv_accuracy['experiment_name']} ({top_cv_accuracy['cv_accuracy']:.4f})\n"
        f"Best by Cross Validation Precision: {top_cv_precision['experiment_name']} ({top_cv_precision['cv_precision']:.4f})\n"
        f"Best by Cross Validation F1: {top_cv_f1['experiment_name']} ({top_cv_f1['cv_f1']:.4f})\n\n"
        f"CONFUSION MATRIX:\n"
        f"Best by TP: {top_tp['experiment_name']} ({top_tp['test_tp']})\n"
        f"Best by TN: {top_tn['experiment_name']} ({top_tn['test_tn']})\n"
        f"Best by FP: {top_fp['experiment_name']} ({top_fp['test_fp']})\n"
        f"Best by FN: {top_fn['experiment_name']} ({top_fn['test_fn']})\n"
    )
    summary_path = os.path.join(summaries, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_txt)
