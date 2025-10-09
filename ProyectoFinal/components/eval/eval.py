# import argparse
# import pandas as pd
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report,
# )
# import json
# import os


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--scored_data", type=str, required=True)
#     parser.add_argument("--eval_output", type=str, required=True)
#     args = parser.parse_args()

#     # Cargar datos con etiquetas reales y predichas
#     df = pd.read_csv(args.scored_data)

#     if "True_Label" not in df.columns or "Predicted_Label" not in df.columns:
#         raise ValueError(
#             "❌ El archivo no contiene columnas 'True_Label' o 'Predicted_Label'."
#         )

#     y_true = df["True_Label"]
#     y_pred = df["Predicted_Label"]

#     # Calcular métricas principales
#     acc = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
#     recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
#     f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
#     cm = confusion_matrix(y_true, y_pred)
#     report = classification_report(y_true, y_pred, output_dict=True)

#     # Crear carpeta de salida si no existe
#     os.makedirs(args.eval_output, exist_ok=True)

#     # Guardar métricas en JSON
#     metrics = {
#         "accuracy": acc,
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1,
#         "confusion_matrix": cm.tolist(),
#     }

#     metrics_path = os.path.join(args.eval_output, "metrics_summary.json")
#     with open(metrics_path, "w") as f:
#         json.dump(metrics, f, indent=4)

#     # Guardar reporte detallado
#     report_path = os.path.join(args.eval_output, "classification_report.json")
#     with open(report_path, "w") as f:
#         json.dump(report, f, indent=4)

#     print("✅ Evaluación completada.")
#     print(f"🔹 Accuracy: {acc:.3f}")
#     print(f"🔹 Precision: {precision:.3f}")
#     print(f"🔹 Recall: {recall:.3f}")
#     print(f"🔹 F1 Score: {f1:.3f}")


# if __name__ == "__main__":
#     main()

import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import json
import os


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

    # ======================================================
    # 🧩 NUEVO BLOQUE: Normalización de tipos (clave del error)
    # ======================================================
    # Asegurar que ambos vectores tengan el mismo tipo (todo numérico o todo texto)
    try:
        # Intentar convertir a numéricos si es posible
        y_true_numeric = pd.to_numeric(y_true, errors="coerce")
        y_pred_numeric = pd.to_numeric(y_pred, errors="coerce")

        if y_true_numeric.notna().all() and y_pred_numeric.notna().all():
            # Si ambos se pudieron convertir correctamente
            y_true = y_true_numeric.astype(int)
            y_pred = y_pred_numeric.astype(int)
            print("🔢 Etiquetas convertidas correctamente a numéricas.")
        else:
            # Si alguna no se pudo convertir, forzamos a string
            y_true = y_true.astype(str)
            y_pred = y_pred.astype(str)
            print("🔤 Etiquetas convertidas a tipo texto para comparación segura.")

    except Exception as e:
        print(f"⚠️ Error al normalizar etiquetas: {e}")
        y_true = y_true.astype(str)
        y_pred = y_pred.astype(str)

    # ======================================================
    # Calcular métricas principales
    # ======================================================
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # Crear carpeta de salida si no existe
    os.makedirs(args.eval_output, exist_ok=True)

    # Guardar métricas en JSON
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = os.path.join(args.eval_output, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Guardar reporte detallado
    report_path = os.path.join(args.eval_output, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print("✅ Evaluación completada.")
    print(f"🔹 Accuracy: {acc:.3f}")
    print(f"🔹 Precision: {precision:.3f}")
    print(f"🔹 Recall: {recall:.3f}")
    print(f"🔹 F1 Score: {f1:.3f}")


if __name__ == "__main__":
    main()
