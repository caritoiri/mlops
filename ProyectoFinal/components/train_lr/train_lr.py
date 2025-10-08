import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    # Cargar datos de entrenamiento
    df = pd.read_csv(args.train)

    # Separar X y y
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Inicializar MLflow
    mlflow.start_run()

    # Definir modelo multiclase
    model = LogisticRegression(max_iter=200, multi_class="multinomial", solver="lbfgs")

    # Entrenar modelo
    model.fit(X, y)

    # Predicciones sobre train
    y_pred = model.predict(X)

    # Calcular métricas básicas
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    cm = confusion_matrix(y, y_pred)

    # Registrar métricas en MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_param("algorithm", "LogisticRegression")

    # Guardar matriz de confusión como archivo de texto
    with open("confusion_matrix.txt", "w") as f:
        f.write(str(cm))

    # Registrar modelo en MLflow
    mlflow.sklearn.log_model(model, "model")

    # Guardar modelo localmente para el siguiente paso del pipeline
    joblib.dump(model, f"{args.model_output}/model.pkl")

    mlflow.end_run()

    print("✅ Entrenamiento completado con éxito.")
    print(f"🔹 Accuracy: {acc:.3f}")
    print(f"🔹 F1 Score: {f1:.3f}")


if __name__ == "__main__":
    main()
