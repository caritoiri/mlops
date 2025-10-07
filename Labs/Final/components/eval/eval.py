import argparse, json
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    y_true, y_pred, y_prob = df["y_true"], df["y_pred"], df["y_prob"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    mlflow.log_artifact(args.out)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.2f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig("pr_curve.png")
    mlflow.log_artifact("pr_curve.png")


if __name__ == "__main__":
    mlflow.start_run(nested=True)
    main()
    mlflow.end_run()
