import argparse, pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.encoded)
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df.drop(columns=["target"])),
        columns=[c for c in df.columns if c != "target"],
    )
    scaled["target"] = df["target"].values
    scaled.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
