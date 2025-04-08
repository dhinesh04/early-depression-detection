import csv
import os
import pandas as pd
from collections import defaultdict
import random
import json
import pdb

csv_folder_path = "csv_output"

def prepare_dataset_for_model(file, label):
    file_path = os.path.join(csv_folder_path, file)
    df = pd.read_csv(file_path)
    df = df[df["TEXT"].notna() & df["TEXT"].str.strip().ne("")]

    # Group by ID
    grouped = df.groupby("ID")["TEXT"].apply(list).reset_index()

    # Convert to structured list of dicts
    return [{"id": row["ID"], "texts": row["TEXT"], "label": label} for _, row in grouped.iterrows()]

def load_data(csv_folder_path):
    for file in os.listdir(csv_folder_path):
        if "2017" in file:
            if "neg" in file:
                neg_data = prepare_dataset_for_model(file, label=0)
                print("Generated neg data")
            else:
                pos_data = prepare_dataset_for_model(file, label=1)
                print("Generated pos data")
    full_dataset = neg_data + pos_data
    print("Merged both data")
    random.shuffle(full_dataset)
    print("Randomized the data")
    full_path = "full_dataset.json"
    with open(full_path, "w") as f:
        json.dump(full_dataset, f, indent=2)
    

def main():
   # Loads dataset for training model
   load_data(csv_folder_path)


if __name__ == "__main__":
    main()

