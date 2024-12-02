import mlflow
import mlflow.tracking 
from mlflow.tracking import MlflowClient
import dagshub
import os
import yaml


#mlflow.set_tracking_uri("https://dagshub.com/sudhir649/mlops-v2-gha-demo.mlflow")  # Adjust this URI as needed
secrets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/secrets.yaml"))
print(f"Loading configuration from: {secrets_path}")

with open(secrets_path, "r") as file:
    secrets = yaml.safe_load(file)

# Extract secrets 

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print("Base Directory:", base_dir)

# Define paths relative to the base directory


best_model_path = os.path.join(base_dir, "artifacts/Best_Model/")

# Ensure your token is correctly set in the environment
os.environ['MLFLOW_TRACKING_TOKEN'] = secrets["mlflowsecrets"]["MLFLOW_TRACKING_TOKEN"]

# Initialize DagsHub's MLflow tracking integration
dagshub.init(repo_owner='sudhir649', repo_name='MlOpsFlask', mlflow=True)

def model_fetching(best_model_path,experiment_name="Default"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    print(experiment)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    experiment_id = experiment.experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName = 'Model Evaluation'"
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
        print(f"Best run ID: {best_run_id} with MSE: {best_MSE} and R2: {best_r2_score}")
        
        # Use artifact_path instead of path
        local_model_path = mlflow.artifacts.download_artifacts(
            run_id=best_run_id,
            artifact_path="model",
            dst_path=best_model_path
        )
        
        print(f"Model downloaded to: {local_model_path}")
    else:
        print("No suitable run found.")

# Usage
model_fetching(best_model_path=best_model_path)

