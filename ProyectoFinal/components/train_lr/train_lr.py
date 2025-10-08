import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    # Cargar datos de entrenamiento
    df = pd.read_csv(args.train)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Entrenar modelo
    model = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)

    # Predicciones
    y_pred = model.predict(X)

    # Calcular métricas
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    cm = confusion_matrix(y, y_pred)

    # Guardar métricas como JSON
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }
    os.makedirs(args.model_output, exist_ok=True)
    with open(os.path.join(args.model_output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Guardar modelo
    joblib.dump(model, os.path.join(args.model_output, "model.pkl"))

    print("✅ Modelo entrenado correctamente.")
    print("🔹 Métricas:", metrics)

if __name__ == "__main__":
    main()
