import re
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import glob
import pandas as pd
import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import random
from competition_chatgpt import get_comp_text
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()


def get_bert_scores(queries_to_docnos, docs_dict, queries_dict):
    scores_dict = {}
    for qid in queries_to_docnos:
        for docno in queries_to_docnos[qid]:
            query_text = queries_dict[qid]
            doc_text = docs_dict[docno]
            ret = tokenizer.encode_plus(query_text,
                                        doc_text,
                                        max_length=512,
                                        truncation=True,
                                        return_token_type_ids=True,
                                        return_tensors='pt')

            with torch.cuda.amp.autocast(enabled=False):
                input_ids = ret['input_ids'].to(device)
                tt_ids = ret['token_type_ids'].to(device)
                output, = model(input_ids, token_type_ids=tt_ids, return_dict=False)
                if output.size(1) > 1:
                    score = torch.nn.functional.log_softmax(output, 1)[0, -1].item()
                else:
                    score = output.item()

            scores_dict[docno] = score

    return scores_dict


def create_data_structures(df, queries_list, test=True):
    df = df[df.query_string.isin(queries_list)]
    queries_dict = dict(zip(df.query_id, df.query_string))
    if test:
        docs_dict = dict()
        queries_to_docnos = {k: ["TEST"] for k in set(df.query_id)}
    else:
        docs_dict = dict(zip(df.docno, df.current_document))
        queries_to_docnos = {k: list(set(df[df.query_id == k].docno)) for k in set(df.query_id)}
    return queries_dict, docs_dict, queries_to_docnos


def get_unique_words(string):
    cleaned_string = re.sub(r'[^\w\s]', '', string.lower())
    words = cleaned_string.split()
    unique_words = set(words)
    stop_words = set(stopwords.words('english'))
    unique_words = unique_words - stop_words
    return unique_words


def get_messages(query):
    # version 7
    messages = [
       {"role": "system",
        "content": fr"You participate in a search engine optimization competition regarding the following topic\s: {query}"},
       {"role": "user",
        "content": f"Generate a single text that a ranker will deem as highly relevant to {query}, incorporate the words {get_unique_words(query)} and avoid stop words in your text as much as possible."
                   " The text should be comprehensive, informative and coherent. Avoid meta commentary.\nText:"}]
    return messages


def generate_text(query, weights):
    return get_comp_text(messages=get_messages(query), temperature=round(float(weights[0]), 2),
                         top_p=round(float(weights[1]), 2),
                         frequency_penalty=round(float(weights[2]), 2), presence_penalty=round(float(weights[3]), 2))


def calculate_similarity(query, queries_to_docnos, docs_dict, queries_dict, idx):
    queries_to_docnos_ = {idx: queries_to_docnos[idx]}
    return get_bert_scores(queries_to_docnos_, docs_dict, queries_dict)


# def plot_histogram(values, save_path):
#     plt.hist(values, bins='auto', color='darkgrey', alpha=0.7, rwidth=0.85)
#     plt.axvline(np.mean(values), color='magenta', linestyle='solid', linewidth=2,
#                 label=f'Mean --> {round(np.mean(values), 3)}')
#     plt.axvline(np.median(values), color='cyan', linestyle='dashed', linewidth=2,
#                 label=f'Median --> {round(np.median(values), 3)}')
#     plt.xlabel('Values')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Values')
#     plt.legend()
#     plt.savefig(f"fine_tuning_files/{save_path}.jpeg", format='jpeg')
#     plt.show()


def save_dictionary(dictionary, save_path):
    with open(f"fine_tuning_files/{save_path}.txt", 'w', encoding='utf-8') as file:
        json.dump(dictionary, file, ensure_ascii=False)


def get_next_version(file_name):
    pattern = rf'{file_name}(\d+)'
    version_numbers = [int(re.search(pattern, f).group(1)) for f in os.listdir("./fine_tuning_files") if
                       re.search(pattern, f)]
    max_version = max(version_numbers) if version_numbers else 0
    return max_version + 1


def get_queries_string(df):
    for idx, row in df.iterrows():
        df.loc[idx, "query_string"] = ", ".join([row.query1, row.query2, row.query3])
    return df


# scoring experiments
if __name__ == '__main__':
    # fixed weights
    weights = [0.2, 0.3, 1.0, 0.0]


    all_files = glob.glob(os.path.join("./data_snapshots/", "*.csv"))
    # check if a temp file exists
    if os.path.exists("./fine_tuning_files/temp_df.csv"):
        df = pd.read_csv("./fine_tuning_files/temp_df.csv")
    else:
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        df = get_queries_string(df)
        df["gen_text"] = np.nan
    queries_list = list(set(df.query_string))
    queries_dict, docs_dict, queries_to_docnos = create_data_structures(
        df[df.round_no == 1].drop_duplicates(subset=['current_document']), queries_list=queries_list)
    idx_dict = {v: k for k, v in queries_dict.items()}
    messages = get_messages(queries_list[0])

    scores_dict = dict()
    scores_dict["messages"] = messages
    scores_dict["weights"] = weights
    scores_dict["data"] = dict()
    # scores_list = []
    for query in tqdm(queries_list):

        # # do not create new text for queries that already have a text
        # if df[df.query_string == query].gen_text.notna().sum() != 0:
        #   continue

        text = generate_text(query, weights)
        # text = "test doc"
        docs_dict["TEST"] = text
        idx = idx_dict[query]
        query_list = [q.strip() for q in query.split(",")]
        temp_queries_dict = queries_dict.copy()
        temp_score_list = []
        for q in query_list:
            temp_queries_dict[idx] = q
            score = calculate_similarity(q, queries_to_docnos, docs_dict, temp_queries_dict, idx)
            temp_score_list.append(score["TEST"])

        # score = calculate_similarity(query, queries_to_docnos, docs_dict, queries_dict)
        scores_dict["data"][idx] = {"score1": temp_score_list[0], "score2": temp_score_list[1],
                                    "score3": temp_score_list[2], "text": text, "query1": query_list[0],
                                    "query2": query_list[1], "query3": query_list[2]}

        # scores_list.append(temp_score_list)
        df["gen_text"][df.query_string == query] = text
        df.to_csv("./fine_tuning_files/temp_df.csv", index=False)

    version = get_next_version('version_')
    save_path = f'version_{version}'
    # plot_histogram(scores_list, save_path)
    save_dictionary(scores_dict, save_path)
    x = 1
