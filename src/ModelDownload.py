import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
secrets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/secrets.yaml"))
logger.info(f"Loading configuration from: {secrets_path}")

with open(secrets_path, "r") as file:
    secrets = yaml.safe_load(file)

# Set environment variables for authentication
os.environ['MLFLOW_TRACKING_TOKEN'] = secrets["mlflowsecrets"]["MLFLOW_TRACKING_TOKEN"]

# Initialize DagsHub integration
dagshub.init(repo_owner='sudhir649', repo_name='MlOpsFlask', mlflow=True)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
logger.info(f"Base Directory: {base_dir}")
best_model_path = os.path.join(base_dir, "artifacts/Best_Model/")
os.makedirs(best_model_path, exist_ok=True)

def download_artifact(run_id, artifact_path, dst_path):
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=dst_path
    )
    return local_path

def model_fetching(best_model_path, experiment_name="Default"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    logger.info(f"Experiment details: {experiment}")

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    experiment_id = experiment.experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName = 'Model_Evaluation'"
    )

    best_MSE = float("inf")
    best_r2_score = float("-inf")
    best_run = None

    for run in runs:
        metrics = run.data.metrics
        mse = metrics.get('mean_squared_error', float('inf'))
        r2_score = metrics.get('r2_score', float('-inf'))

        if mse < best_MSE and r2_score > best_r2_score:
            best_MSE = mse
            best_r2_score = r2_score
            best_run = run

    if best_run:
        best_run_id = best_run.info.run_id
        logger.info(f"Best run ID: {best_run_id} with MSE: {best_MSE} and R2: {best_r2_score}")
        local_model_path = download_artifact(best_run_id, "best_model", best_model_path)
        logger.info(f"Model downloaded to: {local_model_path}")
    else:
        logger.warning("No suitable run found.")

# Run the model fetching process
model_fetching(best_model_path=best_model_path)
