import argparse, pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.scaled)
    train, test = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=42
    )
    train.to_csv(args.train, index=False)
    test.to_csv(args.test, index=False)


if __name__ == "__main__":
    main()
