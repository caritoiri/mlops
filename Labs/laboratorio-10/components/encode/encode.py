import argparse, pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputed", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.imputed)
    df = pd.get_dummies(df, drop_first=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
