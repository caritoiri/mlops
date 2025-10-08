import argparse
import pandas as pd
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--scored_output", type=str, required=True)
    args = parser.parse_args()

    # Cargar modelo entrenado
    model_path = f"{args.model_input}/model.pkl"
    model = joblib.load(model_path)

    # Cargar datos de prueba
    df = pd.read_csv(args.test_data)

    if "Target" not in df.columns:
        raise ValueError(
            "❌ La columna 'Target' no fue encontrada en el dataset de prueba."
        )

    X_test = df.drop(columns=["Target"])
    y_true = df["Target"]

    # Generar predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Crear DataFrame con resultados
    scored_df = X_test.copy()
    scored_df["True_Label"] = y_true
    scored_df["Predicted_Label"] = y_pred

    # Agregar probabilidades si existen
    for i, cls in enumerate(model.classes_):
        scored_df[f"Prob_{cls}"] = y_proba[:, i]

    # Guardar resultados
    scored_df.to_csv(args.scored_output, index=False)
    print(f"✅ Predicciones generadas exitosamente. Archivo: {args.scored_output}")


if __name__ == "__main__":
    main()
