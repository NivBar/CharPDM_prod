import pandas as pd
import glob
import os
import warnings
import openai
import tiktoken
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
import json
import itertools
import config
from tqdm import tqdm

warnings.filterwarnings("ignore")
encoder = tiktoken.encoding_for_model(config.model)


def get_comp_text(messages, temperature, top_p, frequency_penalty, presence_penalty):
    max_tokens = config.max_tokens
    response = False
    prompt_tokens = len(encoder.encode("".join([line['content'] for line in messages]))) + 200
    while prompt_tokens + max_tokens > 4096:
        max_tokens -= 50
        print("Changed max tokens for response to:", max_tokens)

    while not response:
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            # print("success")
            word_no = len(response['choices'][0]['message']['content'].split())
            if word_no > 150:
                max_tokens -= 50
                response = False
                print(f"word no was: {word_no}, dropping max tokens to: {max_tokens}.")
                continue
            break
        except Exception as e:
            print(e)
            continue
    return response


all_files = glob.glob(os.path.join("./data_snapshots/", "*.csv"))
df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)[
    ['round_no', 'query_id', 'group', 'username', 'position1', 'position2', 'position3', 'posted_document']]
df[["position1", "position2", "position3"]] = df[["position1", "position2", "position3"]].astype(int) / 10
df["med_score"] = df[["position1", "position2", "position3"]].median(axis=1)
df["mean_score"] = df[["position1", "position2", "position3"]].mean(axis=1)

keys_ = list(df.groupby(['round_no', 'query_id', 'group']).size().index)
tops = []
for round_no_, query_id_, group_ in set(keys_):
    top = df[(df.round_no == round_no_) & (df.query_id == query_id_) & (df.group == group_)]
    top = top[top.med_score == top.med_score.min()]
    # if len(top) > 1:
    #     top = top[top.mean_score == top.mean_score.min()]
    # if len(top) > 1:
    #     top = top[top.username != "dummy"]
    # if len(top) > 1:
    #     top = top[~top.username.str.contains("BOT")]
    # if len(top) > 1:
    #     top = top.head(1)
    tops.append(top)

topic_data = json.load(open(fr'./data_snapshots/initials.json', encoding="utf8"))
topic_df = pd.DataFrame(topic_data)[["_id", "queries"]]
topic_df._id = topic_df._id.astype(int)

tops_df = pd.concat(tops).sort_values(["round_no", "query_id", "group"])
tops_df = tops_df.merge(topic_df, how="left", right_on="_id", left_on="query_id")
# data = tops_df[["round_no", "query_id", "group", "posted_document", "queries"]].to_dict('records')

##### fine-tuning process: #####

param_grid = {
    'temperature': [0.2, 0.6, 1.0],
    'top_p': [0.3, 0.6, 0.9],
    'frequency_penalty': [0.0, 0.5, 1.0],
    'presence_penalty': [0.0, 0.5, 1.0]
}

combinations = list(itertools.product(*param_grid.values()))
combinations = [dict(zip(param_grid.keys(), combination)) for combination in combinations]

try:
    ft_df = pd.read_csv(fr'fine_tuning_files/fine_tuning_{tops_df.round_no.max()}.csv', encoding="utf8")
except:
    ft_df = pd.DataFrame(
        columns=["temperature", "top_p", "frequency_penalty", "presence_penalty", "round_no", "query_id",
                 "group", "posted_document", "queries", "suggested_document"])
    ft_df.to_csv(fr'fine_tuning_files/fine_tuning_{tops_df.round_no.max()}.csv', index=False, encoding="utf8")

for comb in tqdm(combinations, position=0, total=len(combinations)):
    for query_id in tqdm(tops_df.query_id.unique(), position=1, total=len(tops_df.query_id.unique()), leave=False):
        if not ft_df[(ft_df.query_id == query_id) & (ft_df.temperature == comb["temperature"]) & (
                ft_df.top_p == comb["top_p"]) & (ft_df.frequency_penalty == comb["frequency_penalty"]) & (
                             ft_df.presence_penalty == comb["presence_penalty"])].empty:
            continue

        query_df = tops_df[tops_df.query_id == query_id][
            ["round_no", "query_id", "group", "posted_document", "queries"]].reset_index(drop=True)
        for k, v in comb.items():
            query_df[k] = v

        queries_str = ", ".join(query_df.loc[0, "queries"])

        messages = [
            {"role": "system",
             "content": fr"You participate in a search engine optimization competition regarding the following queries:"
                        fr" {queries_str}. Your document should be ranked first (1st) in relevance to all queries."},
            {"role": "user",
             "content": "Generate a single text that addresses the three queries in a comprehensive, informative and "
                        "coherent manner.\nText:"}]

        res = get_comp_text(messages, temperature=comb["temperature"], top_p=comb["top_p"],
                            frequency_penalty=comb["frequency_penalty"], presence_penalty=comb["presence_penalty"])[
            'choices'][0]['message']['content']
        deleted_segment = min(
            [x for x in [res.split(".")[-1], res.split("!")[-1], res.split("?")[-1], res.split("\n\n")[-1]] if
             x != res], key=len)
        res = res.replace(deleted_segment, "")

        query_df["suggested_document"] = res
        ft_df = pd.concat([ft_df, query_df], ignore_index=True)
        ft_df.to_csv(fr'fine_tuning_files/fine_tuning_{tops_df.round_no.max()}.csv', index=False, encoding="utf8")
