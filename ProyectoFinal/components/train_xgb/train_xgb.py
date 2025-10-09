# ===============================
# train_xgb.py
# ===============================
import argparse
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import joblib, json, os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    # =====================================================
    # 1️⃣ CARGAR DATOS
    # =====================================================
    df = pd.read_csv(args.train)
    print(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")

    if "Target" not in df.columns:
        raise ValueError("❌ No se encontró la columna 'Target' en el dataset de entrenamiento.")

    # =====================================================
    # 2️⃣ LIMPIEZA Y MAPEO DE ETIQUETAS
    # =====================================================
    label_map = {"Dropout": 0, "Enrolled": 1, "Graduated": 2}
    reverse_map = {v: k for k, v in label_map.items()}

    df["Target"] = df["Target"].map(label_map)
    print(f"🔢 Etiquetas mapeadas: {label_map}")

    missing_targets = df["Target"].isna().sum()
    if missing_targets > 0:
        print(f"⚠️ {missing_targets} filas eliminadas por Target vacío")
        df = df.dropna(subset=["Target"])

    X = df.drop(columns=["Target"])
    y = df["Target"].astype(int)

    # =====================================================
    # 3️⃣ VERIFICAR CLASES Y BALANCE
    # =====================================================
    unique_classes = sorted(y.unique())
    print(f"🔍 Clases presentes en el dataset: {unique_classes}")

    for cls in [0, 1, 2]:
        if cls not in unique_classes:
            print(f"⚠️ Clase {cls} ausente, agregando muestra sintética.")
            X.loc[len(X)] = X.mean()
            y.loc[len(y)] = cls

    # Calcular pesos por clase (balanceo)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    weights_dict = {i: w for i, w in enumerate(class_weights)}
    sample_weights = y.map(weights_dict)
    print(f"⚖️ Pesos de clase calculados: {weights_dict}")

    # =====================================================
    # 4️⃣ ENTRENAR MODELO XGBOOST OPTIMIZADO
    # =====================================================
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y, sample_weight=sample_weights)
    print("✅ Entrenamiento XGBoost completado con balance de clases.")

    # =====================================================
    # 5️⃣ EVALUACIÓN SOBRE ENTRENAMIENTO
    # =====================================================
    y_pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    report = classification_report(
        y, y_pred, target_names=list(label_map.keys()), output_dict=True
    )

    # =====================================================
    # 6️⃣ GUARDAR MODELO Y MÉTRICAS
    # =====================================================
    os.makedirs(args.model_output, exist_ok=True)

    with open(os.path.join(args.model_output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(args.model_output, "report.json"), "w") as f:
        json.dump(report, f, indent=4)

    joblib.dump(model, os.path.join(args.model_output, "model.pkl"))

    print("✅ Modelo XGBoost multiclase guardado correctamente.")
    print("📊 Métricas de desempeño:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
