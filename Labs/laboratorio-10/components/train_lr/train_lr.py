import argparse, os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--penalty", type=str, default="l2")  # <-- nuevo
    parser.add_argument("--C", type=float, default=1.0)  # <-- nuevo
    args = parser.parse_args()

    # Cargar datos
    df = pd.read_csv(args.train)
    X, y = df.drop(columns=["target"]), df["target"]

    # Entrenar modelo con los hiperparámetros recibidos
    model = LogisticRegression(max_iter=1000, penalty=args.penalty, C=args.C)
    model.fit(X, y)

    # Calcular accuracy en entrenamiento
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    # Guardar modelo en disco
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.pkl"))

    # Logging en MLflow
    mlflow.log_param("penalty", args.penalty)
    mlflow.log_param("C", args.C)
    mlflow.log_metric("train_accuracy", acc)
    mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    mlflow.start_run(nested=True)
    main()
    mlflow.end_run()
