# components/split/split.py
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df["y"]
    )
    os.makedirs(os.path.dirname(args.train), exist_ok=True)
    train_df.to_csv(args.train, index=False)
    test_df.to_csv(args.test, index=False)


if __name__ == "__main__":
    main()
