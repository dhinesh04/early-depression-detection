**About the Project**

This project named "NLP for Early Mental Health Risk Assessment" is part of the CSE5525 - Foundations of Speech and Language Processing course at The Ohio State University

Language is a powerful indicator of personality, social or emotional status, but also mental health. This project aims to develop an NLP system that can analyze text data from Reddit posts and tweets to predict early signs of mental health disorders and possibly assess the severity of the depression.

**Dataset Used**

In the eRisk 2025 challenge 2, there are two tasks: Task 1 - Search for Symptoms of Depression and
Task 2 - Contextualized Early Detection of Depression. For our project, we will focus on Task 2 of the challenge.

To acquire the dataset used, please contact the organizers of eRisk 2025. - https://erisk.irlab.org/

The data set provided is a collection of Reddit posts for three separate years - 2017, 2018 and 2022 and has two classifications of the data:

• The ”pos” users are individuals who confirmed a diagnosis of depression

• The ”neg” users are control users, meaning they were randomly sampled from Reddit.

****How to Run this project?****

**Tools Used:**
Python 3.9+, A100 GPU for faster training

**Steps:**
1. Create a virtual environment using python3 -m venv myenv
2. Activate using source myenv/bin/activate
3. Install requirements using pip install -r requirements
4. Run xml_to_json.py for parsing XML files
5. Run json_to_csv.py for cleaning the raw data and converting it to csv format
6. Run prepare_data.py to prepare the model for training
7. Run analysis.py to analyse the data and provide various metrics such as TF-IDF and log odds ratio to understand the data clearly
8. Run model_bert_base.py, model_deberta.py, model_long.py and model_pubmed.py for 5 epochs to get the results.
