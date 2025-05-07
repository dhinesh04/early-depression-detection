import os
import csv
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

csv_folder_path = "csv_output"

# =====================
# UTILITY FUNCTIONS
# =====================
def prepare_dataset_for_model(file, label):
    file_path = os.path.join(csv_folder_path, file)
    df = pd.read_csv(file_path)
    df = df[df["TEXT"].notna() & df["TEXT"].str.strip().ne("")]
    grouped = df.groupby("ID")["TEXT"].apply(list).reset_index()
    return [{"id": row["ID"], "texts": row["TEXT"], "label": label} for _, row in grouped.iterrows()]

def load_year_data(year):
    pos_file = f"{year}_cases_pos_xmls.csv"
    neg_file = f"{year}_cases_neg_xmls.csv"
    pos_data = prepare_dataset_for_model(pos_file, label=1)
    neg_data = prepare_dataset_for_model(neg_file, label=0)
    print(f"Loaded {len(pos_data)} positive and {len(neg_data)} negative samples for {year}")
    return pos_data + neg_data

def save_dataset(dataset, filename):
    random.shuffle(dataset)
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} samples to {filename}")

def oversample_minority_class(data, minority_label=1, multiplier=3):
    pos_samples = [d for d in data if d["label"] == minority_label]
    neg_samples = [d for d in data if d["label"] != minority_label]
    oversampled_pos = random.choices(pos_samples, k=len(pos_samples) * multiplier)
    combined = neg_samples + oversampled_pos
    random.shuffle(combined)
    return combined

# =====================
# 1. CLASSIC STRATIFIED SPLIT
# =====================
def classic_stratified_split():
    all_data = load_year_data("2017") + load_year_data("2018") + load_year_data("2022")
    labels = [entry["label"] for entry in all_data]
    train_data, test_data = train_test_split(all_data, test_size=0.2, stratify=labels, random_state=42)
    os.makedirs("prepared_classic_json", exist_ok=True)
    save_dataset(train_data, "prepared_classic_json/train_dataset.json")
    save_dataset(test_data, "prepared_classic_json/test_dataset.json")
    save_dataset(all_data, "prepared_classic_json/full_dataset.json")

# =====================
# 2. STRATIFIED SPLIT + OVERSAMPLING
# =====================
def stratified_split_with_oversampling():
    all_data = load_year_data("2017") + load_year_data("2018") + load_year_data("2022")
    labels = [entry["label"] for entry in all_data]
    train_data, test_data = train_test_split(all_data, test_size=0.2, stratify=labels, random_state=42)
    train_data = oversample_minority_class(train_data, minority_label=1, multiplier=3)
    os.makedirs("prepared_balanced_json", exist_ok=True)
    save_dataset(train_data, "prepared_balanced_json/train_dataset.json")
    save_dataset(test_data, "prepared_balanced_json/test_dataset.json")
    save_dataset(train_data + test_data, "prepared_balanced_json/full_dataset.json")

# =====================
# 3. TEMPORAL EVALUATION
# =====================
def temporal_split_with_oversampling():
    train_data = load_year_data("2017") + load_year_data("2018")
    test_data = load_year_data("2022")
    train_data = oversample_minority_class(train_data, minority_label=1, multiplier=3)
    os.makedirs("prepared_temporal_json", exist_ok=True)
    save_dataset(train_data, "prepared_temporal_json/train_dataset.json")
    save_dataset(test_data, "prepared_temporal_json/test_dataset.json")
    save_dataset(train_data + test_data, "prepared_temporal_json/full_dataset.json")

# =====================
# MAIN
# =====================
def main():
    print("\nRunning Classic Stratified Split...")
    classic_stratified_split()

    print("\nRunning Stratified Split with Oversampling...")
    stratified_split_with_oversampling()

    print("\nRunning Temporal Split with Oversampling...")
    temporal_split_with_oversampling()

if __name__ == "__main__":
    main()
