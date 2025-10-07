import argparse, os, joblib, mlflow
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)  # CSV de predicciones
    args = ap.parse_args()

    pipe = joblib.load(os.path.join(args.model_dir, "model.pkl"))
    df = pd.read_csv(args.test)
    X, y = df.drop(columns=["y"]), df["y"]

    y_prob = pipe.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    out = pd.DataFrame({"y_true": y, "y_pred": y_pred, "y_prob": y_prob})
    out.to_csv(args.out, index=False)
    mlflow.log_artifact(args.out)


if __name__ == "__main__":
    mlflow.start_run(nested=True)
    main()
    mlflow.end_run()
