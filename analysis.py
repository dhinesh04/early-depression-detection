import json
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from empath import Empath

# Load dataset
with open("full_dataset.json") as f:
    raw_data = json.load(f)

# Flatten JSON data
rows = []
for user in raw_data:
    combined_text = " ".join(user["texts"])
    rows.append({"id": user["id"], "label": user["label"], "text": combined_text})

df = pd.DataFrame(rows)

# Split by class
class_0 = df[df['label'] == 0]['text']
class_1 = df[df['label'] == 1]['text']

# ========== TF-IDF ANALYSIS ==========
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
corpus = class_0.tolist() + class_1.tolist()
labels = [0] * len(class_0) + [1] * len(class_1)
tfidf_matrix = vectorizer.fit_transform(corpus)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df['label'] = labels

avg_0 = tfidf_df[tfidf_df['label'] == 0].mean()
avg_1 = tfidf_df[tfidf_df['label'] == 1].mean()
tfidf_diff = (avg_1 - avg_0).drop("label").sort_values(ascending=False)

top_tfidf_pos = tfidf_diff.head(20)
top_tfidf_neg = tfidf_diff.tail(20)

# ========== LOG-ODDS RATIO ==========
def get_word_counts(text_series):
    return Counter(" ".join(text_series).lower().split())

counts_0 = get_word_counts(class_0)
counts_1 = get_word_counts(class_1)
vocab = set(counts_0) | set(counts_1)
alpha = 1

log_odds = {}
for word in vocab:
    a = counts_1[word] + alpha
    b = counts_0[word] + alpha
    log_odds[word] = np.log(a / b)

sorted_log_odds = sorted(log_odds.items(), key=lambda x: x[1], reverse=True)
log_odds_pos = sorted_log_odds[:20]
log_odds_neg = sorted_log_odds[-20:]

# ========== EMPATH (LIWC-style) ==========
lexicon = Empath()
df["empath"] = df["text"].apply(lambda x: lexicon.analyze(x, normalize=True))
empath_df = pd.json_normalize(df["empath"])
empath_df["label"] = df["label"]

mean_0 = empath_df[empath_df['label'] == 0].mean()
mean_1 = empath_df[empath_df['label'] == 1].mean()
empath_diff = (mean_1 - mean_0).sort_values(ascending=False)

empath_pos = empath_diff.head(10)
empath_neg = empath_diff.tail(10)

# Output everything in a dictionary and save as JSON
analysis_result = {
    "top_tfidf_terms_class_1": top_tfidf_pos.to_dict(),
    "top_tfidf_terms_class_0": top_tfidf_neg.to_dict(),
    "log_odds_top_words_class_1": dict(log_odds_pos),
    "log_odds_top_words_class_0": dict(log_odds_neg),
    "empath_categories_class_1": empath_pos.to_dict(),
    "empath_categories_class_0": empath_neg.to_dict()
}

output_path = "linguistic_analysis_results.json"
with open(output_path, "w") as f:
    json.dump(analysis_result, f, indent=2)

# output_path


import json
import matplotlib.pyplot as plt

# Load analysis results again
with open("linguistic_analysis_results.json") as f:
    data = json.load(f)

# Function to plot and save a bar chart
def plot_bar_chart(data_dict, title, filename, color):
    items = list(data_dict.items())
    keys, values = zip(*items)

    plt.figure(figsize=(12, 6))
    plt.bar(keys, values, color=color)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"data/{filename}")
    plt.close()

# TF-IDF Bar Charts
plot_bar_chart(data["top_tfidf_terms_class_1"], "Top TF-IDF Terms - Class 1 (Depressed)", "tfidf_class_1.png", "red")
plot_bar_chart(data["top_tfidf_terms_class_0"], "Top TF-IDF Terms - Class 0 (Control)", "tfidf_class_0.png", "green")

# Log-Odds Ratio Bar Charts
plot_bar_chart(data["log_odds_top_words_class_1"], "Top Log-Odds Words - Class 1 (Depressed)", "logodds_class_1.png", "orange")
plot_bar_chart(data["log_odds_top_words_class_0"], "Top Log-Odds Words - Class 0 (Control)", "logodds_class_0.png", "blue")

# Empath Categories Bar Charts
plot_bar_chart(data["empath_categories_class_1"], "Top Empath Categories - Class 1 (Depressed)", "empath_class_1.png", "purple")
plot_bar_chart(data["empath_categories_class_0"], "Top Empath Categories - Class 0 (Control)", "empath_class_0.png", "gray")


import pandas as pd

# Prepare data for TF-IDF overlay
tfidf_pos = pd.Series(data["top_tfidf_terms_class_1"], name="class_1")
tfidf_neg = pd.Series(data["top_tfidf_terms_class_0"], name="class_0")

# Take top absolute terms from both
top_n = 20
tfidf_all = pd.concat([
    tfidf_pos.abs().sort_values(ascending=False).head(top_n),
    tfidf_neg.abs().sort_values(ascending=False).head(top_n)
])
common_words = list(set(tfidf_all.index))

# Create overlay DataFrame
overlay_df = pd.DataFrame({
    "Class 1 (Depressed)": tfidf_pos.get(common_words),
    "Class 0 (Control)": tfidf_neg.get(common_words)
}, index=common_words).fillna(0)

overlay_df = overlay_df.sort_values(by="Class 1 (Depressed)", ascending=False)

# Plot and save the overlay chart
plt.figure(figsize=(14, 7))
overlay_df.plot(kind="bar", width=0.8, figsize=(14, 7), color=["red", "green"])
plt.title("Overlay of TF-IDF Term Differences: Class 1 vs Class 0")
plt.xticks(rotation=45, ha='right')
plt.ylabel("TF-IDF Score Difference")
plt.tight_layout()
plt.savefig("data/tfidf_overlay_chart.png")
plt.close()

# Return file paths
[
    "data/tfidf_class_1.png",
    "data/tfidf_class_0.png",
    "data/logodds_class_1.png",
    "data/logodds_class_0.png",
    "data/empath_class_1.png",
    "data/empath_class_0.png",
    "data/tfidf_overlay_chart.png"
]

