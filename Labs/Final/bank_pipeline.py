from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.entities import Environment
from dotenv import load_dotenv
import os

load_dotenv()

# Conexión al workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.environ["AZ_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZ_RESOURCE_GROUP"],
    workspace_name=os.environ["AZ_ML_WORKSPACE"],
)

COMPUTE = "cpu-cluster"

# Registrar o recuperar environment asset
env_asset = Environment(
    name="bank-env",
    version="2",
    description="Entorno para Bank Marketing pipelines",
    conda_file="envs/custom_env.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
env_asset = ml_client.environments.create_or_update(env_asset)

# Componentes con nombre
ingest_clean = command(
    name="ingest-clean-step",  # <- nombre explícito
    display_name="Ingest & Clean Data",
    code="components/ingest_clean",
    command="python ingest_clean.py --raw ${{inputs.raw}} --out ${{outputs.clean}}",
    inputs={"raw": Input(type="uri_file")},
    outputs={"clean": Output(type="uri_file")},
    environment=env_asset,
    compute=COMPUTE,
)

split = command(
    name="train_test_split-step",
    display_name="Train/Test Split",
    code="components/split",
    command="python split.py --data ${{inputs.data}} --train ${{outputs.train}} --test ${{outputs.test}}",
    inputs={"data": Input(type="uri_file")},
    outputs={"train": Output(type="uri_file"), "test": Output(type="uri_file")},
    environment=env_asset,
    compute=COMPUTE,
)

train_lr = command(
    name="train_logistic_regression-step",
    display_name="Train LR",
    code="components/train_lr",
    command="python train.py --train ${{inputs.train}} --model_dir ${{outputs.model_dir}} --C 1.0 --penalty l2",
    inputs={"train": Input(type="uri_file")},
    outputs={"model_dir": Output(type="uri_folder")},
    environment=env_asset,
    compute=COMPUTE,
)

train_rf = command(
    name="train_random_forest-step",
    display_name="Train RF",
    code="components/train_rf",
    command="python train.py --train ${{inputs.train}} --model_dir ${{outputs.model_dir}} --n_estimators 300 --max_depth 8",
    inputs={"train": Input(type="uri_file")},
    outputs={"model_dir": Output(type="uri_folder")},
    environment=env_asset,
    compute=COMPUTE,
)

score_lr = command(
    name="score_lr-step",
    display_name="Score LR",
    code="components/score",
    command="python score.py --model_dir ${{inputs.model_dir}} --test ${{inputs.test}} --out ${{outputs.pred}}",
    inputs={"model_dir": Input(type="uri_folder"), "test": Input(type="uri_file")},
    outputs={"pred": Output(type="uri_file")},
    environment=env_asset,
    compute=COMPUTE,
)

score_rf = command(
    name="score_rf-step",
    display_name="Score RF",
    code="components/score",
    command="python score.py --model_dir ${{inputs.model_dir}} --test ${{inputs.test}} --out ${{outputs.pred}}",
    inputs={"model_dir": Input(type="uri_folder"), "test": Input(type="uri_file")},
    outputs={"pred": Output(type="uri_file")},
    environment=env_asset,
    compute=COMPUTE,
)

eval_lr = command(
    name="eval_lr-step",
    display_name="Evaluate LR",
    code="components/eval",
    command="python eval.py --predictions ${{inputs.pred}} --out ${{outputs.metrics}}",
    inputs={"pred": Input(type="uri_file")},
    outputs={"metrics": Output(type="uri_file")},
    environment=env_asset,
    compute=COMPUTE,
)

eval_rf = command(
    name="eval_rf-step",
    display_name="Evaluate RF",
    code="components/eval",
    command="python eval.py --predictions ${{inputs.pred}} --out ${{outputs.metrics}}",
    inputs={"pred": Input(type="uri_file")},
    outputs={"metrics": Output(type="uri_file")},
    environment=env_asset,
    compute=COMPUTE,
)

from azure.ai.ml.dsl import pipeline


@pipeline(default_compute=COMPUTE)
def bank_pipeline(raw_csv):
    a = ingest_clean(raw=raw_csv)
    b = split(data=a.outputs.clean)

    lr = train_lr(train=b.outputs.train)
    rf = train_rf(train=b.outputs.train)

    lr_pred = score_lr(model_dir=lr.outputs.model_dir, test=b.outputs.test)
    rf_pred = score_rf(model_dir=rf.outputs.model_dir, test=b.outputs.test)

    lr_eval = eval_lr(pred=lr_pred.outputs.pred)
    rf_eval = eval_rf(pred=rf_pred.outputs.pred)

    return {
        "lr_metrics": lr_eval.outputs.metrics,
        "rf_metrics": rf_eval.outputs.metrics,
        "lr_model": lr.outputs.model_dir,
        "rf_model": rf.outputs.model_dir,
    }


job = bank_pipeline(raw_csv=Input(type="uri_file", path="data/bank.csv"))
job = ml_client.jobs.create_or_update(
    job,
    experiment_name="bank_pipeline_v2",
    tags={"lab": "10", "dataset": "bank"},
    display_name="BankPipelineFull",
)
print("Submitted pipeline job:", job.name)
