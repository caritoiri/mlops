from azure.ai.ml import MLClient, dsl, Input, Output, load_component
from azure.ai.ml.entities import PipelineJob, Model
from azure.identity import DefaultAzureCredential
import os

# Inicializar conexión al Workspace de Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.getenv("AZ_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZ_RG_NAME"),
    workspace_name=os.getenv("AZ_WS_NAME"),
)

select_cols_comp = load_component(source="components/select_cols/select_cols.yml")
impute_comp = load_component(source="components/impute/impute.yml")
encode_comp = load_component(source="components/encode/encode.yml")
split_comp = load_component(source="components/split/split.yml")
train_comp = load_component(source="components/train_xgb/train_xgb.yml")
# train_comp = load_component(source="components/train_rf/train_rf.yml")
# train_comp = load_component(source="components/train_lr/train_lr.yml")
score_comp = load_component(source="components/score/score.yml")
eval_comp = load_component(source="components/eval/eval.yml")
register_model_comp = load_component(source="components/register_model/register_model.yml")



@dsl.pipeline(
    compute="cpu-cluster",  # Nombre de tu cluster de cómputo en Azure ML
    description="Pipeline de predicción de abandono estudiantil usando ML",
)
def dropout_pipeline(raw_data: Input):

    select_step = select_cols_comp(data=raw_data)
    impute_step = impute_comp(data=select_step.outputs.out)
    encode_step = encode_comp(data=impute_step.outputs.out)
    split_step = split_comp(data=encode_step.outputs.out)
    train_step = train_comp(train=split_step.outputs.train)
    score_step = score_comp(
        model_input=train_step.outputs.model_output, test_data=split_step.outputs.test
    )
    eval_step = eval_comp(scored_data=score_step.outputs.scored_output)
    register_step = register_model_comp(
        model_input=train_step.outputs.model_output,
        eval_metrics=eval_step.outputs.eval_output,
        AZ_SUBSCRIPTION_ID=os.getenv("AZ_SUBSCRIPTION_ID"),
        AZ_RG_NAME=os.getenv("AZ_RG_NAME"),
        AZ_WS_NAME=os.getenv("AZ_WS_NAME")
    )
    return {
        "trained_model": train_step.outputs.model_output,
        "evaluation_report": eval_step.outputs.eval_output,
        "model_registration": register_step.outputs.register_ok,
    }


# Crear pipeline con entrada de datos
pipeline_job = dropout_pipeline(raw_data=Input(path="data/data.csv", type="uri_file"))

# Enviar pipeline al workspace
pipeline_run = ml_client.jobs.create_or_update(pipeline_job)
print(f"🚀 Pipeline enviado correctamente. ID de ejecución: {pipeline_run.name}")