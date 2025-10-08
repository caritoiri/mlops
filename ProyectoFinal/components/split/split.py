import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    args = parser.parse_args()

    # Cargar dataset preprocesado (ya numérico)
    df = pd.read_csv(args.data)

    # Verificar existencia de variable objetivo
    if "Target" not in df.columns:
        raise ValueError("❌ La columna 'Target' no fue encontrada en el dataset.")

    # Dividir en X (features) y y (target)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Dividir en entrenamiento y prueba (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Unir nuevamente para guardar
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Guardar datasets
    train_df.to_csv(args.train, index=False)
    test_df.to_csv(args.test, index=False)

    print(f"✅ División completada:\nTrain: {train_df.shape}\nTest: {test_df.shape}")


if __name__ == "__main__":
    main()
