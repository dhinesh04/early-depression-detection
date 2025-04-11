import json
import csv
import os
import pdb
import re
csv_folder = "csv_output"
os.makedirs(csv_folder, exist_ok=True)
folder_path = "json_output"
os.makedirs(folder_path, exist_ok=True)
json_folder = os.listdir("json_output")
csv_header = ["ID", "TITLE", "DATE", "INFO", "TEXT"]

def clean_text(text):
    if text is None:
        return ""
    text = text.replace("\n", " ")

    # Remove markdown links: [text](url)
    text = re.sub(r'\[.*?\]\(https?://[^\)]+\)', '', text)

    # Remove plain URLs (https or http)
    text = re.sub(r'https?://\S+', '', text)

    # Remove things like `url=...` (used in some Google redirect links)
    text = re.sub(r'url=https?%3A%2F%2F\S+', '', text)

    # Remove remaining URL artifacts and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    tokens = text.lower().split()
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]

    return " ".join(tokens).strip()
    # return text.strip()

def iterate_csv(csv_folder):
    folder_list = os.listdir(csv_folder)
    for file in folder_list:
        csv_file_path = os.path.join(csv_folder, file)
        with open(csv_file_path, "r", encoding="utf-8") as f:
            unique_ids = set()
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[0] in unique_ids:
                    continue
                unique_ids.add(row[0])
        print(f"Number of unique IDs after cleaning for file {csv_file_path} is {len(unique_ids)}")

def convert_to_csv():
    for file in json_folder:
        csv_file_path = os.path.join(csv_folder, os.path.splitext(file)[0] + ".csv")
        with open(os.path.join(folder_path,file), "r") as fle:
            individuals = json.load(fle)
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header
            writer.writerow(csv_header)
            
            # Iterate through individuals
            for individual_data in individuals:
                individual = individual_data["INDIVIDUAL"]
                individual_id = individual["ID"]
                writings = individual["WRITING"]

                # Iterate through writings
                for entry in writings:
                    if entry["TEXT"] == "":
                        continue
                    try:
                        writer.writerow([
                            individual_id, 
                            clean_text(entry["TITLE"]),
                            clean_text(entry["DATE"]),
                            clean_text(entry["INFO"]),
                            clean_text(entry["TEXT"])])
                    except:
                        continue
        print(f"CSV file saved successfully: {csv_file_path}")

convert_to_csv()
iterate_csv(csv_folder)