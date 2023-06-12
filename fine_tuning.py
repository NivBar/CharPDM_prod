from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.stats import percentileofscore as pos
from tqdm import tqdm
import os
import glob
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

# def get_bert_scores(queries_to_docnos, docs_dict, queries_dict):
#     scores_dict = {}
#     for qid in tqdm(queries_to_docnos, desc="BERT", total=len(queries_to_docnos)):
#         for docno in queries_to_docnos[qid]:
#             query_text = queries_dict[qid]
#             doc_text = docs_dict[docno]
#             ret = tokenizer.encode_plus(query_text,
#                                         doc_text,
#                                         max_length=512,
#                                         truncation=True,
#                                         return_token_type_ids=True,
#                                         return_tensors='pt')
#             with torch.cuda.amp.autocast(enabled=False):
#                 input_ids = ret['input_ids'].to(device)
#                 tt_ids = ret['token_type_ids'].to(device)
#                 output, = model(input_ids, token_type_ids=tt_ids, return_dict=False)
#                 if output.size(1) > 1:
#                     score = torch.nn.functional.log_softmax(output, 1)[0, -1].item()
#                 else:
#                     score = output.item()
#             scores_dict[docno] = score
#     return scores_dict

# def create_data_structures(df,col):
#     queries_dict = dict(zip(df.query_id, df[col]))
#     docs_dict = dict(zip(df.docno, df.current_document))
#     queries_to_docnos = {k: list(set(df[df.query_id == k].docno)) for k in set(df.query_id)}
#     return queries_dict, docs_dict, queries_to_docnos

# all_files = glob.glob(os.path.join("./data_snapshots/", "data_snapshots_*.csv"))
# df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

df = pd.read_csv("./data_snapshots/BERT_scores_5.csv")
with open("./fine_tuning_files/version_9.txt", 'r', encoding='utf-8') as file:
    test_data = json.loads(file.read())
test_df = pd.DataFrame(test_data["data"]).T.reset_index().rename({"index": "query_id", "text": "current_document"},
                                                                 axis=1)
test_df["username"] = "TEST"
concat_df = pd.concat([df, test_df])
for col in ["query_id", "score1", "score2", "score3"]:
    concat_df[col] = concat_df[col].astype(float)


def get_stats(concat_df, ranker):
    print(f"Ranker: {ranker}")
    if ranker == "LambdaMART":
        concat_df = concat_df[concat_df.group.isin(["A", "B"]) | (concat_df.username == "TEST")]
    elif ranker == "BERT":
        concat_df = concat_df[concat_df.group.isin(["C", "D"]) | (concat_df.username == "TEST")]

    pos_dict = {1: [], 2: [], 3: [], "avg": [], "med": []}

    for query in concat_df.query_id.unique():
        # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        avg_pos = 0
        med_pos = []
        for i in range(3):
            temp_df = concat_df[(concat_df.query_id == query) & (concat_df[f"score{i + 1}"].notna())]
            try:
                test_score = list(temp_df[temp_df.username == "TEST"][f"score{i + 1}"])[0]
            except:
                x = 1
            score_values = list(temp_df[(temp_df.username != "TEST")][f"score{i + 1}"])
            # sns.histplot(score_values, bins=10, ax=axes[i])
            # axes[i].set(title=f'{query} {i+1} - Histogram', xlabel='Score', ylabel='Amount of Scores')
            # axes[i].scatter(test_score, 0, s=150, color='green' if test_score > max(score_values) else 'red')
            # axes[i].legend([f'Test Score, pos: {pos(score_values, test_score):.2f}%'], loc='upper left')
            pos_dict[i + 1].append(pos(score_values, test_score))
            avg_pos += pos(score_values, test_score)
            med_pos.append(pos(score_values, test_score))
        pos_dict["avg"].append(avg_pos / 3)
        pos_dict["med"].append(np.median(med_pos))

        # plt.tight_layout()
        # plt.show()

    print({k: np.round(np.mean(v),3) for k, v in pos_dict.items()})


if __name__ == '__main__':
    get_stats(concat_df, "LambdaMART")
    get_stats(concat_df, "BERT")
    x = 1
