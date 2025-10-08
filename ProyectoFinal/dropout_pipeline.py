from azure.ai.ml import MLClient, dsl, Input, Output
from azure.ai.ml.entities import PipelineJob
from azure.identity import DefaultAzureCredential

# Inicializar conexión al Workspace de Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="YOUR_SUBSCRIPTION_ID",
    resource_group_name="YOUR_RESOURCE_GROUP",
    workspace_name="YOUR_WORKSPACE_NAME",
)

# Referencias a los componentes (ya registrados o en el repositorio local)
select_cols_comp = ml_client.components.get("select_cols")
impute_comp = ml_client.components.get("impute")
encode_comp = ml_client.components.get("encode")
split_comp = ml_client.components.get("split")
train_comp = ml_client.components.get("train_lr")
score_comp = ml_client.components.get("score")
eval_comp = ml_client.components.get("eval")


@dsl.pipeline(
    compute="cpu-cluster",  # Nombre de tu cluster de cómputo en Azure ML
    description="Pipeline de predicción de abandono estudiantil usando ML",
)
def dropout_pipeline(raw_data: Input):

    # 1️⃣ Selección de variables relevantes
    select_step = select_cols_comp(data=raw_data)

    # 2️⃣ Imputación de valores faltantes
    impute_step = impute_comp(data=select_step.outputs.out)

    # 3️⃣ Codificación categórica
    encode_step = encode_comp(data=impute_step.outputs.out)

    # 4️⃣ División train/test
    split_step = split_comp(data=encode_step.outputs.out)

    # 5️⃣ Entrenamiento del modelo
    train_step = train_comp(train=split_step.outputs.train)

    # 6️⃣ Generación de predicciones
    score_step = score_comp(
        model_input=train_step.outputs.model_output, test_data=split_step.outputs.test
    )

    # 7️⃣ Evaluación de métricas finales
    eval_step = eval_comp(scored_data=score_step.outputs.scored_output)

    # Salida final del pipeline
    return {
        "trained_model": train_step.outputs.model_output,
        "evaluation_report": eval_step.outputs.eval_output,
    }


# Crear pipeline con entrada de datos
pipeline_job = dropout_pipeline(raw_data=Input(path="data/data.csv", type="uri_file"))

# Enviar pipeline al workspace
pipeline_run = ml_client.jobs.create_or_update(pipeline_job)
print(f"🚀 Pipeline enviado correctamente. ID de ejecución: {pipeline_run.name}")
