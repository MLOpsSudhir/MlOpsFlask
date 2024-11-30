import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error ,r2_score, mean_absolute_error
import os
import argparse
import mlflow




base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print("Base Directory:", base_dir)

# Define paths relative to the base directory

model_data_path = os.path.join(base_dir, "artifacts/model/")
processed_data_path = os.path.join(base_dir, "artifacts/data/processed_data/")
model_saving_path = os.path.join(base_dir, "artifacts/model_evaluation/")

def model_evaluation(model_data_path,processed_data_path,model_saving_path):

    model_path_file_name = os.path.join(model_data_path,"my_model.pkl")

    model = pickle.load(open(model_path_file_name,"rb"))

    X_test_path = os.path.join(processed_data_path,"X_test.csv")
    Y_test_path = os.path.join(processed_data_path,"Y_test.csv")

    X_test = pd.read_csv(X_test_path)

    Y_test = pd.read_csv(Y_test_path)

    Y_pred_test = model.predict(X_test)

    r_score = r2_score(Y_test,Y_pred_test)

    mse = mean_squared_error(Y_test,Y_pred_test)

    mbe = mean_absolute_error(Y_test,Y_pred_test)

    print(r_score)

    print(mse)

    print(mbe)



    #if r_score > .4:
    print(r_score)
    model_saving_path_file = os.path.join(model_saving_path,"perfect_model.pkl")
    pickle.dump(model,open(model_saving_path_file,"wb"))
        # Log the model in MLflow
    #mlflow.log_artifact(model_saving_path_file, artifact_path="model_evaluation")
        
        # Log metrics in MLflow
    #mlflow.log_metric("r2_score", r_score)
   # mlflow.log_metric("mean_squared_error", mse)
    #mlflow.log_metric("mean_absolute_error", mbe)
    mlflow.log_metrics({"r2_score":r_score,
                        "mean_squared_error":mse,
                        "mean_absolute_error":mbe })

        # Optionally, you can log the model with MLflow as well (for tracking)
    mlflow.sklearn.log_model(model, "best_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_data_path" , help="provide model data path" , default=model_data_path)
    parser.add_argument("--processed_data_path" , help="provide proceesed data path" , default=processed_data_path)
    parser.add_argument("--model_saving_path" , help="provide model path data path" , default=model_saving_path)
    
    args = parser.parse_args()
    model_evaluation(args.model_data_path,args.processed_data_path,args.model_saving_path)