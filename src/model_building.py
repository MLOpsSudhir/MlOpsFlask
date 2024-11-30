import pandas as pd
import pickle
import os
import argparse
import mlflow

from sklearn.linear_model import LinearRegression


#model_data_path = "artifacts/model/"
#processed_data_path = "artifacts/data/processed_data/"


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print("Base Directory:", base_dir)

# Define paths relative to the base directory

model_data_path = os.path.join(base_dir, "artifacts/model/")
processed_data_path = os.path.join(base_dir, "artifacts/data/processed_data/")


def model_building(model_data_path,processed_data_path):
    print("######################mlflow model Building Started +++++++++++")
    X_train_path = os.path.join(processed_data_path,"X_train.csv")
    Y_train_path = os.path.join(processed_data_path,"Y_train.csv")
    
    X_train = pd.read_csv(X_train_path)

    Y_train = pd.read_csv(Y_train_path)

    #model = LinearRegression(fit_intercept=True)
    model = LinearRegression(fit_intercept=False)
    model_data_path_file_name = os.path.join(model_data_path,"my_model.pkl")
    model.fit(X_train,Y_train)
    pickle.dump(model,open(model_data_path_file_name,"wb"))
    mlflow.log_param("mode_path",model_data_path_file_name)
    mlflow.sklearn.log_model(model,"my_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_data_path" , help="provide model data path" , default=model_data_path)
    parser.add_argument("--processed_data_path" , help="provide proceesed data path" , default=processed_data_path)
    args = parser.parse_args()
    model_building(args.model_data_path,args.processed_data_path)
        



