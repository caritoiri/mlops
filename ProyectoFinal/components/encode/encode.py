import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Cargar dataset limpio desde componente 'impute'
    df = pd.read_csv(args.data)

    # Detectar columnas categóricas (tipo object o category)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Evitar codificar la variable objetivo 'Target'
    cat_cols = [c for c in cat_cols if c != "Target"]

    # Aplicar One-Hot Encoding
    if cat_cols:
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        df_encoded = df.copy()

    # Asegurar que no existan valores NaN después del encoding
    df_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_encoded.fillna(0, inplace=True)

    # Guardar resultado
    df_encoded.to_csv(args.out, index=False)
    print(f"✅ Codificación completada. Archivo generado en: {args.out}")


if __name__ == "__main__":
    main()
