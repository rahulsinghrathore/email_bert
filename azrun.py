from azureml.core.model import Model
from azureml.core.workspace import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.core.script_run_config import ScriptRunConfig
from azureml.core.environment import Environment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import DockerConfiguration



ws = Workspace(
    subscription_id = "88ffd436-6b2f-4a5c-942f-72cc66d5bfab",
    resource_group = "aarav_resources",
    workspace_name = "aarav_workspace",
)

ct = ComputeTarget(workspace=ws, name="aaravCompute")
# env = Environment.get(workspace=ws, name="tf_env")
env = Environment.from_conda_specification(name='env', file_path='./deployment/ENV.yml')
docker_config = DockerConfiguration(use_docker=False)
src = ScriptRunConfig(
    source_directory=".",
    script="train.py",
    compute_target=ct,
    environment=env,
    docker_runtime_config=docker_config,
)
run = Experiment(workspace=ws, name="email_spam_bert").submit(src)
run.wait_for_completion(show_output=True)