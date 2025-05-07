import xmltodict
import json
import csv
import os
import pdb
from tqdm import tqdm

xml_folder = "eRisk25-datasets/task2-contextualized-early-depression/training_data"
output_folder = "json_output"
years = os.listdir(xml_folder)
os.makedirs(output_folder, exist_ok=True)

for year in years:
    for type in os.listdir(f"{xml_folder}/{year}"):
        # pdb.set_trace()
        json_list = []
        xml_files = os.listdir(f"{xml_folder}/{year}/{type}")
        for xml_file in tqdm(xml_files, desc=f"Iterating through {xml_folder}/{year}/{type}"):
            if xml_file.endswith(".xml"):
                try:
                    with open(f"{xml_folder}/{year}/{type}/{xml_file}", "r", encoding="utf-8") as f:
                        xml_data = f.read()
                        json_data = json.loads(json.dumps(xmltodict.parse(xml_data)))
                        json_list.append(json_data)
                except:
                    pdb.set_trace()
        output_filename = f"{year}_{type}_xmls.json"
        output_path = os.path.join(output_folder, output_filename)    
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(json_list, json_file, indent=4)

        print(f"Saved JSON: {output_path}")