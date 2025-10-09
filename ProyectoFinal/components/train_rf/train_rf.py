# ===============================
# train_rf_multiclass.py
# ===============================
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import joblib
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    # ===============================
    # 1️⃣ Cargar datos de entrenamiento
    # ===============================
    df = pd.read_csv(args.train)
    print(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")

    X = df.drop(columns=["Target"])
    y = df["Target"]

    # ===============================
    # 2️⃣ Convertir variable objetivo a numérica
    # ===============================
    if y.dtype == "object":
        mapping = {label: idx for idx, label in enumerate(y.unique())}
        print(f"🔢 Mapeo de etiquetas: {mapping}")
        y = y.map(mapping)

    # ===============================
    # 3️⃣ Entrenar modelo Random Forest
    # ===============================
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    print("✅ Entrenamiento completado.")

    # ===============================
    # 4️⃣ Predicciones y métricas
    # ===============================
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    cm = confusion_matrix(y, y_pred)

    report = classification_report(y, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    # ===============================
    # 5️⃣ Guardar resultados
    # ===============================
    os.makedirs(args.model_output, exist_ok=True)

    # Guardar métricas
    with open(os.path.join(args.model_output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Guardar modelo
    joblib.dump(model, os.path.join(args.model_output, "model.pkl"))

    print("✅ Modelo Random Forest multiclase entrenado correctamente.")
    print("🔹 Métricas:", metrics)

if __name__ == "__main__":
    main()
