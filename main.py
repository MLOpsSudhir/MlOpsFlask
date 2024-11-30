import argparse
import mlflow
import time
import dagshub
dagshub.init(repo_owner='sudhir649', repo_name='mlops-v2-gha-demo', mlflow=True)

time = int(time.time())
def main():

    with mlflow.start_run(run_name=f"MLOPS_{time}") as run:
        mlflow.run("./src",entry_point="Data_Cleaning.py",env_manager="local",run_name="Data Cleaning")
        mlflow.run("./src",entry_point="Data_preprocessing.py",env_manager="local",run_name="Data Preprocessing")
        mlflow.run("./src",entry_point="model_building.py",env_manager="local",run_name="model Building")
        mlflow.run("./src",entry_point="Model_Evaluation.py",env_manager="local",run_name="Model Evaluation")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
