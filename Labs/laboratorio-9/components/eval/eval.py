import argparse, pandas as pd, json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
