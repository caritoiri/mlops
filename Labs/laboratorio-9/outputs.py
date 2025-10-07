from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="ffc4c015-c8d3-44de-982e-2fb248a57ad7",
    resource_group_name="rg-mlops-ucb",
    workspace_name="ws-mlops-ucb",
)

job_name = "24b6a365-62d5-4062-918b-c944e1ed6c73"
print(f"Descargando outputs del job {job_name}")
ml_client.jobs.download(name=job_name, output_name="metrics", download_path="outputs/")

job_name = "f20eddf4-80ab-4097-b3fe-56c178c58119"
ml_client.jobs.download(
    name=job_name, output_name="predictions", download_path="outputs/"
)

job_name = "c350fdc7-1c16-4606-801d-cefb21f096cd"
ml_client.jobs.download(
    name=job_name, output_name="model_dir", download_path="outputs/"
)

print("Modelo descargado en ./outputs/model_dir/")
print("Archivo guardado en ./outputs/predictions/")
