#!/usr/bin/env python3

import pandas as pd
import argparse


def verificar(path_csv, target="y", sample_size=5):
    # Leer CSV (ajusta separador según versión)
    df = pd.read_csv(path_csv, sep=";")

    print("=== VERIFICACION DATASET: Bank Marketing ===")
    print("Total de registros:", len(df))
    print("Columnas:", list(df.columns))
    print("Distribución target ('{}'):".format(target))
    print(df[target].value_counts(dropna=False))
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("\nMuestra de datos:")
    print(df.head(sample_size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, required=True, help="Ruta al archivo bank.csv"
    )
    parser.add_argument(
        "--sample", type=int, default=5, help="Cantidad de filas de muestra"
    )
    args = parser.parse_args()

    verificar(args.csv, sample_size=args.sample)


if __name__ == "__main__":
    main()
