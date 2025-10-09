# ===============================
# train_xgb_v2_balanced.py
# ===============================
import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib, json, os, numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    # 1️⃣ Cargar datos
    df = pd.read_csv(args.train)
    print(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")

    # 2️⃣ Mapear etiquetas a valores numéricos
    label_map = {"Dropout": 0, "Enrolled": 1, "Graduated": 2}
    df["Target"] = df["Target"].map(label_map)
    missing_targets = df["Target"].isna().sum()
    if missing_targets > 0:
        print(f"⚠️ {missing_targets} filas eliminadas por Target vacío")
        df = df.dropna(subset=["Target"])

    X = df.drop(columns=["Target"])
    y = df["Target"].astype(int)

    # 3️⃣ Balancear clases con SMOTE
    print("⚖️ Aplicando balanceo de clases (SMOTE)...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"🔄 Dataset balanceado: {len(y_res)} muestras totales ({np.bincount(y_res)})")

    # 4️⃣ División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    print(f"🧩 División: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # 5️⃣ Definir modelo base XGBoost
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    # 6️⃣ Búsqueda aleatoria de hiperparámetros
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0, 0.1, 0.2],
        "reg_lambda": [1, 1.5, 2],
    }

    print("🔍 Iniciando búsqueda aleatoria de hiperparámetros...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_weighted",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"🏆 Mejores hiperparámetros: {search.best_params_}")

    # 7️⃣ Evaluación final
    y_pred = best_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    report = classification_report(y_test, y_pred, output_dict=True)
    print("✅ Evaluación completada.")
    print(f"🔹 Accuracy: {metrics['accuracy']:.3f}")
    print(f"🔹 Precision: {metrics['precision']:.3f}")
    print(f"🔹 Recall: {metrics['recall']:.3f}")
    print(f"🔹 F1 Score: {metrics['f1_score']:.3f}")

    # 8️⃣ Guardar resultados
    os.makedirs(args.model_output, exist_ok=True)
    with open(os.path.join(args.model_output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(args.model_output, "report.json"), "w") as f:
        json.dump(report, f, indent=4)

    joblib.dump(best_model, os.path.join(args.model_output, "model.pkl"))
    print("✅ Modelo XGBoost balanceado y optimizado guardado correctamente.")

if __name__ == "__main__":
    main()
