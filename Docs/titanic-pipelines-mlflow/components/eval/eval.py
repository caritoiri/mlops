import argparse, os, json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, average_precision_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt


def _save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xlabel="Predicción",
        ylabel="Real",
        title="Matriz de Confusión",
    )
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_roc_curve(y_true, y_prob, out_path):
    disp = RocCurveDisplay.from_predictions(y_true, y_prob)
    disp.ax_.set_title("Curva ROC")
    plt.tight_layout()
    plt.gcf().set_dpi(120)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(plt.gcf())


def _save_pr_curve(y_true, y_prob, out_path):
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    disp.ax_.set_title("Curva Precision–Recall")
    plt.tight_layout()
    plt.gcf().set_dpi(120)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(plt.gcf())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="evaluate")
    args = parser.parse_args()

    # Asegurar solo la carpeta del archivo de salida (esta sí es escribible)
    metrics_dir = os.path.dirname(args.metrics_out)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    # Cargar predicciones
    df = pd.read_csv(args.predictions)
    assert {"y_true", "y_pred"}.issubset(df.columns), "predictions file must have y_true, y_pred"
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    y_prob = df["y_prob"].values if "y_prob" in df.columns else None

    # Métricas base
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Binario?
    is_binary = len(np.unique(y_true)) == 2

    # Métricas con probabilidades
    if y_prob is not None and is_binary:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    # Confusion matrix como escalares (solo binario)
    if is_binary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    # Series para Metrics (ROC/PR)
    roc_points = {"fpr": [], "tpr": []}
    pr_points  = {"precision": [], "recall": []}
    if y_prob is not None and is_binary:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_points["fpr"] = fpr.tolist()
        roc_points["tpr"] = tpr.tolist()
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_points["precision"] = precision_arr.tolist()
        pr_points["recall"]    = recall_arr.tolist()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # GUARDAR PLOTS EN UN DIRECTORIO LOCAL (NO EN EL DIRECTORIO DEL OUTPUT)
    plots_dir = os.path.abspath(os.path.join(os.getcwd(), "eval_artifacts", "plots"))
    os.makedirs(plots_dir, exist_ok=True)
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    roc_path = os.path.join(plots_dir, "roc_curve.png")
    pr_path  = os.path.join(plots_dir, "pr_curve.png")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Generar gráficos
    _save_confusion_matrix(y_true, y_pred, cm_path)
    if y_prob is not None and is_binary:
        _save_roc_curve(y_true, y_prob, roc_path)
        _save_pr_curve(y_true, y_prob, pr_path)

    # Guardar JSON de métricas en el path de salida (uri_file)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    # Log a MLflow (nested)
    with mlflow.start_run(run_name=args.run_name, nested=True):
        # Escalares
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        # Series ROC
        for i, (f, t) in enumerate(zip(roc_points.get("fpr", []), roc_points.get("tpr", []))):
            mlflow.log_metric("roc_fpr", float(f), step=i)
            mlflow.log_metric("roc_tpr", float(t), step=i)
        # Series PR
        for i, (p, r) in enumerate(zip(pr_points.get("precision", []), pr_points.get("recall", []))):
            mlflow.log_metric("pr_precision", float(p), step=i)
            mlflow.log_metric("pr_recall", float(r), step=i)
        # Artefactos (JSON + imágenes)
        mlflow.log_artifact(args.metrics_out, artifact_path="metrics")
        if os.path.exists(cm_path):  mlflow.log_artifact(cm_path,  artifact_path="plots")
        if os.path.exists(roc_path): mlflow.log_artifact(roc_path, artifact_path="plots")
        if os.path.exists(pr_path):  mlflow.log_artifact(pr_path,  artifact_path="plots")

    print(f"Saved metrics to {args.metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
