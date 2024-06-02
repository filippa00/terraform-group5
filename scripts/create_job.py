from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, Command,CommandComponent, JobResourceConfiguration, AmlCompute
from azure.ai.ml.entities import Model
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import AzureBlobDatastore
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient



# Define the subscription, resource group, and workspace name
subscription_id = "0de120ed-f2b8-4aec-84d1-474c0b8fdc77"
resource_group = "ml-terraform-group5"
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
    name="ml-terraform-environment",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

ml_client.environments.create_or_update(environment)


store = AzureBlobDatastore(
    name="datastorename",
    description="Chess csv",
    account_name="newaccount",
    container_name="newcontainer"
)

ml_client.create_or_update(store)





# Define the command component
command_component = CommandComponent(
    name="train-model-component",
    display_name="Train Model",
    description="Component to train model",
    inputs={},
    outputs={},
    code="train_model.py",  # local path where the training script is stored
    command="python train_model.py",
    environment="azureml:ml-terraform-environment:1"
)


# Define the job
job = Command(
    component=command_component,
    compute=compute_cluster_name,  # the compute target
    resources=JobResourceConfiguration(
        instance_count=1
    ),
    experiment_name="example-experiment",
    display_name="example-job",
    environment=environment
)

# Submit the job
ml_client.jobs.create_or_update(job)

# # Create a Model object
# model = Model(name="Terraform-model", model_content="trained_model.pkl")

# # Register the model
# registered_model = ml_client.models.register(model)