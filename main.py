import argparse
import mlflow
import time
import dagshub
import yaml
import os
import dagshub
# Initialize DagsHub MLflow integration
#dagshub.init(repo_owner='sudhir649', repo_name='mlops-v2-gha-demo', mlflow=True)

current_time = int(time.time())

#mlflow.set_tracking_uri("https://dagshub.com/sudhir649/mlops-v2-gha-demo.mlflow")  # Adjust this URI as needed
secrets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config/secrets.yaml"))
print(f"Loading configuration from: {secrets_path}")

with open(secrets_path, "r") as file:
    secrets = yaml.safe_load(file)

# Extract secrets 



# Ensure your token is correctly set in the environment
os.environ['MLFLOW_TRACKING_TOKEN'] = secrets["mlflowsecrets"]["MLFLOW_TRACKING_TOKEN"]

# Initialize DagsHub's MLflow tracking integration
dagshub.init(repo_owner='sudhir649', repo_name='MlOpsFlask', mlflow=True)




# # Extract secrets and set environment variables
# os.environ['MLFLOW_TRACKING_URI'] = secrets["mlflowsecrets"]["MLFLOW_TRACKING_URI"]
# os.environ['MLFLOW_TRACKING_USERNAME'] = secrets["mlflowsecrets"]["MLFLOW_USER_NAME"]
# os.environ['MLFLOW_TRACKING_PASSWORD'] = secrets["mlflowsecrets"]["MLFLOW_PASSWORD"]



def main():
    try:
        with mlflow.start_run(run_name=f"MLOPS_{current_time}") as run:
            print("Starting MLflow pipeline...")

            # Running different MLflow entry points sequentially
            mlflow.run("./src", entry_point="Data_Cleaning.py", env_manager="local", run_name="Data Cleaning")
            mlflow.run("./src", entry_point="Data_preprocessing.py", env_manager="local", run_name="Data Preprocessing")
            mlflow.run("./src", entry_point="model_building.py", env_manager="local", run_name="Model Building")
            mlflow.run("./src", entry_point="Model_Evaluation.py", env_manager="local", run_name="Model Evaluation")
            
            print("MLflow pipeline completed successfully.")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow pipeline with DagsHub integration.")
    args = parser.parse_args()  # You can remove this if no arguments are needed
    main()
