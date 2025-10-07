# components/train/train.py
import argparse, os, joblib, json, warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import mlflow
import mlflow.sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--register_name", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.test_out), exist_ok=True)

    # === Cargar dataset ===
    df = pd.read_csv(args.data)
    target_col = "target"
    assert target_col in df.columns, f"Column '{target_col}' not found in CSV."

    candidate_features = [c for c in df.columns if c != target_col]
    X = df[candidate_features]
    y = df[target_col]

    # === Preprocesamiento ===
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, candidate_features)]
    )

    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===== MLflow tracking =====
    with mlflow.start_run(run_name="train"):
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])

        # Entrenar
        pipe.fit(X_train, y_train)

        # Guardar modelo local
        model_path = os.path.join(args.model_dir, "model.pkl")
        joblib.dump(pipe, model_path)

        # Loguear modelo en MLflow
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        if args.register_name:
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name=args.register_name,
            )

        # === Calcular métricas ===
        y_pred = pipe.predict(X_test)
        y_prob = (
            pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            except Exception:
                pass

        # Guardar métricas en JSON
        metrics_path = os.path.join(args.model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Log de métricas en MLflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # === Guardar preview de predicciones ===
        preview = X_test.copy()
        preview["y_true"] = y_test.values
        preview["y_pred"] = y_pred
        if y_prob is not None:
            preview["y_prob"] = y_prob
        preview.head(20).to_csv(
            os.path.join(args.model_dir, "pred_preview.csv"), index=False
        )

    # Exportar test completo
    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv(args.test_out, index=False)

    print(f"✅ Modelo guardado en: {model_path}")
    print(f"✅ Métricas guardadas en: {metrics_path}")
    print(f"✅ Test split guardado en: {args.test_out}")


if __name__ == "__main__":
    main()
