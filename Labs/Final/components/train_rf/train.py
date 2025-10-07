import argparse, os, joblib, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score

NUM = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CAT = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    X, y = df.drop(columns=["y"]), df["y"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),  # imputar valores faltantes numéricos
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                [c for c in NUM if c in X.columns],
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),  # imputar categorías faltantes
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [c for c in CAT if c in X.columns],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    pipe.fit(X, y)

    p_prob = pipe.predict_proba(X)[:, 1]
    p_lbl = (p_prob >= 0.5).astype(int)
    acc = accuracy_score(y, p_lbl)
    auc = roc_auc_score(y, p_prob)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.model_dir, "model.pkl"))

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("train_acc", acc)
    mlflow.log_metric("train_auc", auc)
    mlflow.sklearn.log_model(pipe, "model")


if __name__ == "__main__":
    mlflow.start_run(nested=True)
    main()
    mlflow.end_run()
