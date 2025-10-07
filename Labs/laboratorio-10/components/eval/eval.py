import argparse, pandas as pd, json
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Leer predicciones
    df = pd.read_csv(args.predictions)
    y_true, y_pred, y_prob = df["y_true"], df["y_pred"], df["y_prob"]

    # Calcular métricas
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    # Guardar métricas en JSON (como antes)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    # Loguear métricas en MLflow
    for key, val in metrics.items():
        mlflow.log_metric(key, val)

    # --- Curva ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {metrics['roc_auc']:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

    # --- Curva Precision-Recall ---
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("pr_curve.png")
    mlflow.log_artifact("pr_curve.png")


if __name__ == "__main__":
    mlflow.start_run(nested=True)
    main()
    mlflow.end_run()
