# components/ingest_clean/ingest_clean.py
import argparse, os
import pandas as pd
import numpy as np

CANON = ["age","job","marital","education","default","balance","housing","loan",
         "contact","day","month","duration","campaign","pdays","previous","poutcome","y"]

def read_smart_csv(path: str) -> pd.DataFrame:
    # Detectar separador ; o ,
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(path, sep=sep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, required=True)         # CSV bruto (Kaggle/UCI)
    ap.add_argument("--out", type=str, required=True)         # CSV limpio
    args = ap.parse_args()

    df = read_smart_csv(args.raw)
    # Normalizar nombres
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Si hay columnas extra, mantenerlas; si faltan, crear como NaN
    for col in CANON:
        if col not in df.columns:
            df[col] = np.nan

    # Reordenar (las no-canónicas quedan al final)
    others = [c for c in df.columns if c not in CANON]
    df = df[CANON + others]

    # Limpieza básica
    df = df.replace({"unknown": np.nan, "UNKNOWN": np.nan})
    for c in ["age","balance","day","duration","campaign","pdays","previous"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Objetivo a binario
    if "y" in df.columns:
        df["y"] = df["y"].astype(str).str.strip().str.lower().map({"yes":1,"no":0})

    # Duplicados (opcional)
    df = df.drop_duplicates()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()

