import pandas as pd
import os
import argparse
import mlflow

# Get the base directory relative to the script's location
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print("Base Directory:", base_dir)

# Define paths relative to the base directory
raw_data_path = os.path.join(base_dir, "artifacts/data/raw_data/")
Cleaned_data_path = os.path.join(base_dir, "artifacts/data/cleaned_data/")
raw_data_file = "sales.csv"

def genrate_file_name(file_name):
    revised_file_name = file_name.split(".csv")
    revised_file_name.insert(-1, "Clean.csv")
    revised_file_name.pop()
    revised_file_name = "_".join(revised_file_name)
    return revised_file_name

def data_cleaning(raw_data_path, Cleaned_data_path, raw_data_file):
    print("###########mlflow started###############")
    raw_data = os.path.join(raw_data_path, raw_data_file)
    if not os.path.exists(raw_data):
        raise FileNotFoundError(f"The file {raw_data} does not exist.")
    
    df = pd.read_csv(raw_data)
    Cleaned_data_file = genrate_file_name(raw_data_file)
    Cleaned_data = os.path.join(Cleaned_data_path, Cleaned_data_file)
    
    os.makedirs(Cleaned_data_path, exist_ok=True)  # Ensure the directory exists
    df.to_csv(Cleaned_data, index=False)
    mlflow.log_param("Cleaned_data_path", Cleaned_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", help="Provide raw data path", default=raw_data_path)
    parser.add_argument("--Cleaned_data_path", help="Provide processed data path", default=Cleaned_data_path)
    parser.add_argument("--raw_data_file", help="Provide raw data file name", default=raw_data_file)
    args = parser.parse_args()
    data_cleaning(args.raw_data_path, args.Cleaned_data_path, args.raw_data_file)
