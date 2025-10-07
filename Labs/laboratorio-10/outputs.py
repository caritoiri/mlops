from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="ffc4c015-c8d3-44de-982e-2fb248a57ad7",
    resource_group_name="rg-mlops-ucb",
    workspace_name="ws-mlops-ucb",
)

job_name = "eb53f66d-2767-4fa5-b106-0917f4fcf3a3"
print(f"Descargando outputs del job {job_name}")
ml_client.jobs.download(name=job_name, output_name="metrics", download_path="outputs/")

job_name = "eb53f66d-2767-4fa5-b106-0917f4fcf3a3"
ml_client.jobs.download(
    name=job_name, output_name="predictions", download_path="outputs/"
)

job_name = "c350fdc7-1c16-4606-801d-cefb21f096cd"
ml_client.jobs.download(
    name=job_name, output_name="model_dir", download_path="outputs/"
)

print("Modelo descargado en ./outputs/model_dir/")
print("Archivo guardado en ./outputs/predictions/")
