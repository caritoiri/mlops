import argparse
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_data", type=str, required=True)
    parser.add_argument("--eval_output", type=str, required=True)
    args = parser.parse_args()

    # Cargar datos con etiquetas reales y predichas
    df = pd.read_csv(args.scored_data)

    if "True_Label" not in df.columns or "Predicted_Label" not in df.columns:
        raise ValueError(
            "❌ El archivo no contiene columnas 'True_Label' o 'Predicted_Label'."
        )

    y_true = df["True_Label"]
    y_pred = df["Predicted_Label"]

    # Calcular métricas
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # Iniciar MLflow
    mlflow.start_run()

    # Registrar métricas
    mlflow.log_metric("accuracy_test", acc)
    mlflow.log_metric("precision_test", precision)
    mlflow.log_metric("recall_test", recall)
    mlflow.log_metric("f1_test", f1)

    # Registrar matriz de confusión
    mlflow.log_text(str(cm), "confusion_matrix.txt")

    # Guardar reporte en JSON
    report_path = f"{args.eval_output}/metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    mlflow.end_run()

    print("✅ Evaluación completada.")
    print(f"🔹 Accuracy: {acc:.3f}")
    print(f"🔹 Precision: {precision:.3f}")
    print(f"🔹 Recall: {recall:.3f}")
    print(f"🔹 F1 Score: {f1:.3f}")


if __name__ == "__main__":
    main()
