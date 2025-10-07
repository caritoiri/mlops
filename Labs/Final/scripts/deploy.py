import os, joblib, json
import pandas as pd

USE_TA = os.getenv("USE_TEXT_ANALYTICS", "false").lower() == "true"
if USE_TA:
    from azure.ai.textanalytics import TextAnalyticsClient, AzureKeyCredential

    def make_ta_client():
        return TextAnalyticsClient(
            endpoint=os.environ["AZ_TA_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZ_TA_KEY"]),
        )


def init():
    global model, ta_client
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "model.pkl")
    model = joblib.load(model_path)
    ta_client = make_ta_client() if USE_TA else None


def run(raw_data):
    try:
        payload = json.loads(raw_data)
        X = pd.DataFrame(payload["data"])

        if USE_TA and "last_call_notes" in X.columns:
            docs = X["last_call_notes"].fillna("").tolist()
            res = ta_client.analyze_sentiment(docs)
            # score simple: positivo - negativo
            scores = [
                d.confidence_scores.positive - d.confidence_scores.negative for d in res
            ]
            X["ta_sentiment"] = scores
            # opcional: quitar el texto crudo
            # X = X.drop(columns=["last_call_notes"])

        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {"pred": pred.tolist(), "proba": proba.tolist()}
    except Exception as e:
        return {"error": str(e)}
