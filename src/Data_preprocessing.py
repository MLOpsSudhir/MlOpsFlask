import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
import mlflow

#df = pd .read_csv("artifacts/data/raw_data/sales.csv")

#X = df[["Sales"]]
#Y = df ["Advertisment"]

#cleaned_data_path = "artifacts/data/cleaned_data/"
#processed_data_path = "artifacts/data/processed_data"
#Target = "Advertisment"


# Get the base directory relative to the script's location
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print("Base Directory:", base_dir)

# Define paths relative to the base directory

cleaned_data_path = os.path.join(base_dir, "artifacts/data/cleaned_data/")
processed_data_path = os.path.join(base_dir, "artifacts/data/processed_data/")
Target = "Advertisment"






def data_processing(cleaned_data_path, processed_data_path, Target):
    print("#############MLFLOW started for data preprocessing #######")
    cleaned_data = os.listdir(cleaned_data_path)[0]
    cleaned_data = os.path.join(cleaned_data_path,cleaned_data)
    df = pd.read_csv(cleaned_data)
    Y=df[Target]
    X=df.drop(columns=[Target])
    test_size=.4
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    X_train.to_csv(os.path.join(processed_data_path,"X_train.csv"),index=False)
    X_test.to_csv(os.path.join(processed_data_path,"X_test.csv"),index=False)
    Y_train.to_csv(os.path.join(processed_data_path,"Y_train.csv"),index=False)
    Y_test.to_csv(os.path.join(processed_data_path,"Y_test.csv"),index=False)
    mlflow.log_param("Test_Size",test_size)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_data_path" , help="provide clean data path" , default=cleaned_data_path)
    parser.add_argument("--process_data_path" , help="provide proceesed data path" , default=processed_data_path)
    #parser.add_argument("--Target" , help="provide raw data file name" , default=None)
    parser.add_argument("--Target" , help="provide raw data file name" , default=Target)
    args = parser.parse_args()
    if args.Target !=None:
        data_processing(args.clean_data_path,args.process_data_path,args.Target)
    else:
        print(f"Error please provide the {args.Target}")
