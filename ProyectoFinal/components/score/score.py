# import argparse
# import pandas as pd
# import joblib


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_input", type=str, required=True)
#     parser.add_argument("--test_data", type=str, required=True)
#     parser.add_argument("--scored_output", type=str, required=True)
#     args = parser.parse_args()

#     # Cargar modelo entrenado
#     model_path = f"{args.model_input}/model.pkl"
#     model = joblib.load(model_path)

#     # Cargar datos de prueba
#     df = pd.read_csv(args.test_data)

#     if "Target" not in df.columns:
#         raise ValueError(
#             "❌ La columna 'Target' no fue encontrada en el dataset de prueba."
#         )

#     X_test = df.drop(columns=["Target"])
#     y_true = df["Target"]

#     # Generar predicciones
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)

#     # Crear DataFrame con resultados
#     scored_df = X_test.copy()
#     scored_df["True_Label"] = y_true
#     scored_df["Predicted_Label"] = y_pred

#     # Agregar probabilidades si existen
#     for i, cls in enumerate(model.classes_):
#         scored_df[f"Prob_{cls}"] = y_proba[:, i]

#     # Guardar resultados
#     scored_df.to_csv(args.scored_output, index=False)
#     print(f"✅ Predicciones generadas exitosamente. Archivo: {args.scored_output}")


# if __name__ == "__main__":
#     main()


# ===============================
# score.py (versión mejorada)
# ===============================
import argparse
import pandas as pd
import joblib
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--scored_output", type=str, required=True)
    args = parser.parse_args()

    # ===============================
    # 1️⃣ Cargar modelo entrenado
    # ===============================
    model_path = os.path.join(args.model_input, "model.pkl")
    model = joblib.load(model_path)

    # ===============================
    # 2️⃣ Cargar datos de prueba
    # ===============================
    df = pd.read_csv(args.test_data)

    if "Target" not in df.columns:
        raise ValueError("❌ La columna 'Target' no fue encontrada en el dataset de prueba.")

    X_test = df.drop(columns=["Target"])
    y_true = df["Target"]

    # ===============================
    # 3️⃣ Definir mapeo de etiquetas
    # ===============================
    # Usa el mismo orden del entrenamiento
    label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduated"}
    reverse_map = {v: k for k, v in label_map.items()}

    # Si las etiquetas verdaderas son texto, convertirlas a números
    if y_true.dtype == "object":
        y_true_mapped = y_true.map(reverse_map)
    else:
        y_true_mapped = y_true

    # ===============================
    # 4️⃣ Generar predicciones
    # ===============================
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # ===============================
    # 5️⃣ Unificar formato de salida
    # ===============================
    # Convertir todo a texto para que eval.py no tenga conflictos
    y_true_final = pd.Series(y_true_mapped).map(label_map)
    y_pred_final = pd.Series(y_pred).map(label_map)

    # ===============================
    # 6️⃣ Crear DataFrame con resultados
    # ===============================
    scored_df = X_test.copy()
    scored_df["True_Label"] = y_true_final
    scored_df["Predicted_Label"] = y_pred_final

    # Agregar columnas con probabilidades por clase
    for i, cls in enumerate(model.classes_):
        label_name = label_map.get(cls, f"Class_{cls}")
        scored_df[f"Prob_{label_name}"] = y_proba[:, i]

    # ===============================
    # 7️⃣ Guardar resultados
    # ===============================
    os.makedirs(os.path.dirname(args.scored_output), exist_ok=True)
    scored_df.to_csv(args.scored_output, index=False, encoding="utf-8")

    print("✅ Predicciones generadas exitosamente.")
    print(f"📁 Archivo de salida: {args.scored_output}")
    print(f"🧠 Ejemplo de filas:\n{scored_df[['True_Label', 'Predicted_Label']].head()}")

if __name__ == "__main__":
    main()
