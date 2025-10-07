# Imagen base con Miniconda
FROM continuumio/miniconda3:25.3.1-1

# Variables de entorno para no detener la instalación con prompts
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar sistema e instalar utilidades básicas
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y curl wget git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Crear entorno conda para Azure ML
RUN conda create -n azureml python=3.10 -y

# Activar el entorno por defecto en el contenedor
SHELL ["conda", "run", "-n", "azureml", "/bin/bash", "-c"]

# Instalar SDK v2 de Azure ML + dependencias comunes
RUN pip install --no-cache-dir \
    azure-ai-ml \
    azureml-core \
    azureml-pipeline \
    azureml-dataprep \
    azureml-dataset-runtime \
    azureml-defaults \
    numpy pandas scikit-learn matplotlib seaborn jupyter \
    mlflow azureml-train-automl-client

# Instalar Azure CLI + extensión ML
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    az extension add -n ml -y && \
    az extension update -n ml

# Establecer el entorno conda "azureml" como default
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "azureml"]
CMD ["python", "--version"]
