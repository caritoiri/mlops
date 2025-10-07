import argparse, pandas as pd, joblib, os
import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Cargar modelo y dataset
    model = joblib.load(os.path.join(args.model_dir, "model.pkl"))
    df = pd.read_csv(args.test)
    X, y = df.drop(columns=["target"]), df["target"]

    # Predicciones
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Guardar predicciones a CSV
    out = pd.DataFrame({"y_true": y, "y_pred": y_pred, "y_prob": y_prob})
    out.to_csv(args.out, index=False)

    # Registrar artifact en MLflow
    mlflow.log_artifact(args.out)


if __name__ == "__main__":
    mlflow.start_run(nested=True)
    main()
    mlflow.end_run()
