import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Leer dataset
    df = pd.read_csv(args.data)

    # Identificar columnas numéricas y categóricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Imputación de valores numéricos: media
    if num_cols:
        num_imputer = SimpleImputer(strategy="mean")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Imputación de valores categóricos: moda (valor más frecuente)
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Guardar resultado limpio
    df.to_csv(args.out, index=False)
    print(f"✅ Limpieza completada. Datos listos: {args.out}")


if __name__ == "__main__":
    main()
