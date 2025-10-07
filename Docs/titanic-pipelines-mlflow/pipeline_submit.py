import os, json
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input
from azure.ai.ml.entities import PipelineJob, Data
from azure.ai.ml import load_component

# === Configure your workspace ===
SUBSCRIPTION_ID = "d34cfff7-b752-4c2c-88e6-0934466c31a9"
RESOURCE_GROUP  = "rg-mlops-ucb"
WORKSPACE_NAME  = "mlops-ucb"
COMPUTE_NAME    = "cpu-cluster"

# ──────────────────────────────────────────────────────────────────────────────
# 0) Conexión al Workspace
# ──────────────────────────────────────────────────────────────────────────────
print("0) Conectando al Workspace…")
cred = DefaultAzureCredential()
ml_client = MLClient(cred, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

ws = ml_client.workspaces.get(WORKSPACE_NAME)
print(f"   ✔ Conectado a Workspace: {ws.name}")

# ──────────────────────────────────────────────────────────────────────────────
# 1) Carga de componentes (train, score, eval)
# ──────────────────────────────────────────────────────────────────────────────
print("1) Cargando componentes YAML…")
train_comp = load_component(source="components/train/train.yml")
score_comp = load_component(source="components/score/score.yml")
eval_comp  = load_component(source="components/eval/eval.yml")
print("   ✔ Componentes cargados: train, score, eval")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Registro/actualización del Data Asset
# ──────────────────────────────────────────────────────────────────────────────
print("2) Creando/actualizando Data Asset titanic-csv…")
data_asset = Data(
    name="titanic-csv",
    description="CSV Titanic para clases",
    path="data/titanic.csv",   # ruta local; el SDK la sube al datastore del WS
    type="uri_file"
)
created = ml_client.data.create_or_update(data_asset)
print(f"   ✔ Data asset registrado: {created.name} v{created.version}")

data_ref = f"azureml:{created.name}:{created.version}"  # fija versión exacta
register_name = "titanic-logreg-mlflow"  # para el paso de train

# ──────────────────────────────────────────────────────────────────────────────
# 3) Definición del Pipeline (train -> score -> eval)
# ──────────────────────────────────────────────────────────────────────────────
print("3) Definiendo pipeline titanic_pipeline…")
@dsl.pipeline(compute=COMPUTE_NAME, description="Titanic pipeline: train -> score -> eval")
def titanic_pipeline(raw_data: Input):
    train_step = train_comp(raw_data=raw_data, register_name=register_name)
    score_step = score_comp(model_dir=train_step.outputs.model_dir, test_data=train_step.outputs.test_data)
    eval_step  = eval_comp(predictions=score_step.outputs.predictions)
    return {"metrics": eval_step.outputs.metrics, "plots": eval_step.outputs.plots}
print("   ✔ Pipeline definido")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Construcción del job de pipeline
# ──────────────────────────────────────────────────────────────────────────────
print("4) Construyendo objeto PipelineJob…")
pipeline_job: PipelineJob = titanic_pipeline(raw_data=Input(type="uri_file", path=data_ref))
print("   ✔ Objeto PipelineJob creado")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Envío del pipeline a Azure ML
# ──────────────────────────────────────────────────────────────────────────────
print("5) Enviando PipelineJob a Azure ML…")
pipeline_job.display_name = "Titanic-MLflow-Pipeline"
pipeline_job.experiment_name = "titanic-mlflow"   # aparece en Experiments
pipeline_job.tags = {
    "course": "DML-004",
    "module": "MLOps",
    "topic": "MLflow + Pipelines",
}
returned_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"   ✔ Pipeline enviado con éxito: {returned_job.name}")
print(f"   🔗 URL en Azure ML Studio: {returned_job.studio_url}")