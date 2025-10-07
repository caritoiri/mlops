import argparse, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    X, y = df.drop(columns=["target"]), df["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.pkl"))


if __name__ == "__main__":
    main()
