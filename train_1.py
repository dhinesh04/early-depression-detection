import pandas as pd
from collections import defaultdict

# Load the negative and positive datasets
neg_df = pd.read_csv("/Users/tahahaque/Documents/nlpproj/early-depression-detection/csv_output/2017_cases_neg_xmls.csv")
pos_df = pd.read_csv("/Users/tahahaque/Documents/nlpproj/early-depression-detection/csv_output/2017_cases_pos_xmls.csv")

# Add a label column: 0 for non-depressed (neg) and 1 for depressed (pos)
neg_df['label'] = 0
pos_df['label'] = 1

# Concatenate both datasets
df = pd.concat([neg_df, pos_df], ignore_index=True)

# Display the combined dataframe
print(df.head())

grouped_texts = defaultdict(list)

# Iterate through the dataframe and group texts
for idx, row in df.iterrows():
    test_subject = row['ID']
    text = row['TEXT']
    label = row['label']
    grouped_texts[test_subject].append((text, label))

# Convert the defaultdict to a dictionary for easier usage
grouped_texts = dict(grouped_texts)

# Print a sample
for key, value in list(grouped_texts.items())[:3]:
    print(f"Test Subject: {key}")
    for text, label in value:
        print(f"Label: {label}, Text: {text}")
    print("\n")