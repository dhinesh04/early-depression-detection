import json
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import contractions
import nltk
from nltk.corpus import words as nltk_words
import os

nltk.download("words")

# Create folder to save plots
os.makedirs("data", exist_ok=True)

# === Preprocessing Helpers ===
def expand_contractions(text):
    return contractions.fix(text)

def remove_short_words(text, min_length=3):
    return ' '.join([word for word in text.split() if len(word) >= min_length])

def get_word_counts(text_series):
    return Counter(" ".join(text_series).lower().split())

def compute_tfidf_vectors(class_0, class_1):
    extra_stopwords = {
        "ive", "like", "really", "just", "dont", "im", "youre", "id", "didnt",
        "wasnt", "doesnt", "isnt", "wont", "cant", "couldnt", "wouldnt", "shouldnt"
    }
    custom_stopwords = ENGLISH_STOP_WORDS.union(extra_stopwords)

    vectorizer = TfidfVectorizer(stop_words=list(custom_stopwords), max_features=5000)
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
    return top_tfidf_pos, top_tfidf_neg

def compute_log_odds_ratio(class_0, class_1, valid_words):
    medical_terms = {"rexulti", "pristiq", "paxil", "pregabalin", "prozac"}
    valid_words.update(medical_terms)
    counts_0 = get_word_counts(class_0)
    counts_1 = get_word_counts(class_1)
    min_count = 5
    vocab = {
        word for word in set(counts_0) | set(counts_1)
        if (
            (counts_0[word] + counts_1[word]) >= min_count and
            word.isalpha() and
            word in valid_words
        )
    }

    alpha = 1
    log_odds = {}
    for word in vocab:
        a = counts_1[word] + alpha
        b = counts_0[word] + alpha
        log_odds[word] = np.log(a / b)

    sorted_log_odds = sorted(log_odds.items(), key=lambda x: x[1], reverse=True)
    log_odds_pos = sorted_log_odds[:20]
    log_odds_neg = sorted_log_odds[-20:]
    return log_odds_pos, log_odds_neg

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

def main():
    # Load dictionary
    valid_words = set(w.lower() for w in nltk_words.words())

    # Load and preprocess dataset
    with open("prepared_balanced_json/full_dataset.json") as f:
        raw_data = json.load(f)

    rows = []
    for user in raw_data:
        combined_text = " ".join(user["texts"])
        rows.append({"id": user["id"], "label": user["label"], "text": combined_text})
    df = pd.DataFrame(rows)

    df['text'] = df['text'].apply(expand_contractions).str.lower()
    df['text'] = df['text'].apply(remove_short_words)

    class_0 = df[df['label'] == 0]['text']
    class_1 = df[df['label'] == 1]['text']

    # TF-IDF
    top_tfidf_pos, top_tfidf_neg = compute_tfidf_vectors(class_0, class_1)

    # Log-Odds
    log_odds_pos, log_odds_neg = compute_log_odds_ratio(class_0, class_1, valid_words)

    # Save results
    analysis_result = {
        "top_tfidf_terms_class_1": top_tfidf_pos.to_dict(),
        "top_tfidf_terms_class_0": top_tfidf_neg.to_dict(),
        "log_odds_top_words_class_1": dict(log_odds_pos),
        "log_odds_top_words_class_0": dict(log_odds_neg),
    }

    with open("linguistic_analysis_results.json", "w") as f:
        json.dump(analysis_result, f, indent=2)

    # Plot charts
    plot_bar_chart(analysis_result["top_tfidf_terms_class_1"], "Top TF-IDF Terms - Class 1 (Depressed)", "tfidf_class_1.png", "red")
    plot_bar_chart(analysis_result["top_tfidf_terms_class_0"], "Top TF-IDF Terms - Class 0 (Control)", "tfidf_class_0.png", "green")
    plot_bar_chart(analysis_result["log_odds_top_words_class_1"], "Top Log-Odds Words - Class 1 (Depressed)", "logodds_class_1.png", "orange")
    plot_bar_chart(analysis_result["log_odds_top_words_class_0"], "Top Log-Odds Words - Class 0 (Control)", "logodds_class_0.png", "blue")

if __name__ == '__main__':
    main()