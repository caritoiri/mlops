import argparse, os, joblib, json, warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--pred_out", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.pred_out), exist_ok=True)

    # Cargar modelo entrenado
    model_path = os.path.join(args.model_dir, "model.pkl")
    pipe = joblib.load(model_path)

    # Cargar test data
    df_test = pd.read_csv(args.test_data)
    assert "target" in df_test.columns, "Test data must include 'target' column."

    X_test = df_test.drop(columns=["target"])
    y_true = df_test["target"].values

    # Predicciones
    y_prob = None
    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    # Guardar predicciones
    out = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if y_prob is not None:
        out["y_prob"] = y_prob

    out.to_csv(args.pred_out, index=False)

    # Log en MLflow
    with mlflow.start_run(run_name="score", nested=True):
        mlflow.log_artifact(args.pred_out, artifact_path="predictions")

    print(f"Saved predictions to {args.pred_out}")


if __name__ == "__main__":
    main()
