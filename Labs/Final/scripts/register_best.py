import json, os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

SUB, RG, WS = (
    os.environ["AZ_SUBSCRIPTION_ID"],
    os.environ["AZ_RESOURCE_GROUP"],
    os.environ["AZ_ML_WORKSPACE"],
)
ml = MLClient(DefaultAzureCredential(), SUB, RG, WS)

# Rutas a outputs descargados o montados de la ejecución
lr_metrics = json.load(open("artifacts/lr/metrics.json"))
rf_metrics = json.load(open("artifacts/rf/metrics.json"))
winner = (
    ("lr", "artifacts/lr")
    if lr_metrics["roc_auc"] >= rf_metrics["roc_auc"]
    else ("rf", "artifacts/rf")
)
label, path = winner

reg = ml.models.create_or_update(
    Model(
        name="bank-marketing-classifier",
        path=os.path.join(path, "model_dir"),
        description=f"Best model: {label}",
        type="custom_model",
        tags={"lab": "10", "winner": label},
    )
)
print("Registered:", reg.name, reg.version)
