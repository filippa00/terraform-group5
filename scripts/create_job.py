from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, Command,CommandComponent, JobResourceConfiguration, AmlCompute

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
compute_cluster_name = "cpu-cluster"
compute_cluster = AmlCompute(
    name=compute_cluster_name,
    size="STANDARD_D2_V2",
    min_instances=0,
    max_instances=4
)


# Define the environment
environment = Environment(
    name="ml-terraform-environment",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

ml_client.environments.create_or_update(environment)

# Define the command component
command_component = CommandComponent(
    name="train-model-component",
    display_name="Train Model",
    description="Component to train model",
    inputs={},
    outputs={},
    code="scripts",  # local path where the training script is stored
    command="python train_model.py",
    environment="azureml:ml-terraform-environment:1"
)


# Define the job
job = Command(
    component=command_component,
    compute="cpu-cluster",  # the compute target
    resources=JobResourceConfiguration(
        instance_count=1
    ),
    experiment_name="example-experiment",
    display_name="example-job"
)

# Submit the job
ml_client.jobs.create_or_update(job)