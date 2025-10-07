import argparse, pandas as pd, numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.selected)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
