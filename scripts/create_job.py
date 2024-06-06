from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, Command,CommandComponent, JobResourceConfiguration, AmlCompute
from azure.ai.ml.entities import Model
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import AzureBlobDatastore
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import argparse
from azure.ai.ml.constants import AssetTypes
import time

# Define the subscription, resource group, and workspace name
subscription_id = "0de120ed-f2b8-4aec-84d1-474c0b8fdc77"
resource_group = "terraform-group5"
workspace_name = "ml-workspace"

# Create a MLClient
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

# Create or retrieve the compute cluster
compute_cluster_name = "Chess2016-Pipeline"
compute_cluster = AmlCompute(
    name=compute_cluster_name,
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=4
)
# Begin creating or updating the compute cluster
compute_cluster_lro = ml_client.compute.begin_create_or_update(compute_cluster)
compute_cluster = compute_cluster_lro.result()

# Define the environment
environment = Environment(
    name="terraform-environment",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)
ml_client.environments.create_or_update(environment)

# Create or retrieve the compute cluster
compute_cluster_name = "Chess2016-Pipeline"
compute_cluster = AmlCompute(
    name=compute_cluster_name,
    size="STANDARD_D2_V2",
    min_instances=0,
    max_instances=1
)
# Begin creating or updating the compute cluster
compute_cluster_lro = ml_client.compute.begin_create_or_update(compute_cluster)
compute_cluster = compute_cluster_lro.result()

# Define the command component
command_component = CommandComponent(
    name="trainmodel",
    display_name="Train Model",
    description="Component to train model",
      inputs={
        "chess_csv": Input(
            type="uri_file"
        ),
    },
    outputs={},
    code="",  # local path where the training script is stored
    command="python train_model.py --data ${{inputs.chess_csv}}",
    environment="azureml:ml-terraform-environment:1"
)
ml_client.components.create_or_update(command_component)

# Define the job
job = Command(
    component=command_component,
    compute=compute_cluster_name,  # the compute target
    environment="AzureML-lightgbm-3.2-ubuntu18.04-py37-cpu@latest",
    inputs={
        "chess_csv": Input(
            type="uri_file",
            path="2016_CvH.csv",  # Make sure this path is correct
        ),
    },
    resources=JobResourceConfiguration(
        instance_count=1
    ),
    experiment_name="terraform-experiment",
    display_name="terraform-job"
)

returned_job = ml_client.jobs.create_or_update(job)
job_id = returned_job.name
print("${returned_job}")

while True:
    job_status = ml_client.jobs.get(job_id).status
    if job_status in ['Completed', 'Failed', 'Canceled']:
        break
    print(f"Job status: {job_status}")
    time.sleep(30) # reiterate job_status every 10 secs

if job_status != 'Completed':
    raise Exception(f"Job did not complete succesfully: {job_status}")

job = ml_client.jobs.get(job_id)

# Create a Model object
model = Model(
    path=f"azureml://jobs/{job_id}/outputs/artifacts/paths/model/",
    name="Terraform-model",
    description="Model automated through Azure-SDK",
    type=AssetTypes.MLFLOW_MODEL 
)
# Register the model
registered_model = ml_client.models.create_or_update(model)