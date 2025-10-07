from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input
from azure.ai.ml import load_component
from azure.ai.ml.entities import PipelineJob, Data

SUBSCRIPTION_ID = "ffc4c015-c8d3-44de-982e-2fb248a57ad7"
RESOURCE_GROUP = "rg-mlops-ucb"
WORKSPACE_NAME = "ws-mlops-ucb"
COMPUTE_NAME = "cpu-cluster"

cred = DefaultAzureCredential()
ml_client = MLClient(cred, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

# Registrar dataset
data_asset = Data(
    name="heart-disease-csv", path="data/heart-disease.csv", type="uri_file"
)
created = ml_client.data.create_or_update(data_asset)
data_ref = f"azureml:{created.name}:{created.version}"

# Cargar componentes
select_cols = load_component("components/select_cols/select_cols.yml")
impute = load_component("components/impute/impute.yml")
encode = load_component("components/encode/encode.yml")
scale = load_component("components/scale/scale.yml")
split = load_component("components/split/split.yml")
train_lr = load_component("components/train_lr/train_lr.yml")
score = load_component("components/score/score.yml")
evalc = load_component("components/eval/eval.yml")


@dsl.pipeline(compute=COMPUTE_NAME, description="Heart pipeline: 8 steps")
def heart_pipeline(raw: Input):
    s = select_cols(raw_data=raw)
    imp = impute(selected=s.outputs.selected)
    enc = encode(imputed=imp.outputs.imputed)
    sc = scale(encoded=enc.outputs.encoded)
    sp = split(scaled=sc.outputs.scaled)
    tr = train_lr(train=sp.outputs.train)
    sc2 = score(model_dir=tr.outputs.model_dir, test=sp.outputs.test)
    # ev = evalc(predictions=sc2.outputs.out)
    ev = evalc(predictions=sc2.outputs.predictions)

    # return {"metrics": ev.outputs.out, "model": tr.outputs.model_dir}
    return {"metrics": ev.outputs.metrics, "model": tr.outputs.model_dir}


pipeline_job: PipelineJob = heart_pipeline(raw=Input(type="uri_file", path=data_ref))
pipeline_job.display_name = "Heart-Disease-8steps"
pipeline_job.experiment_name = "heart-disease-mlops"
returned_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"Pipeline enviado: {returned_job.name}")
print(f"URL en Azure ML Studio: {returned_job.studio_url}")
