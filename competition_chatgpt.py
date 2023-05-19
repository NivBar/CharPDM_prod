import openai
import config
import tiktoken
import pandas as pd
import glob
from bson.objectid import ObjectId
import warnings

warnings.filterwarnings("ignore")

encoder = tiktoken.encoding_for_model(config.model)


def get_top_user(data, r):
    df = data[data.round_no == r][["username", "position1", "position2", "position3"]].set_index("username")
    return df.median(axis=1).idxmin()


def get_data(epoch=None):
    path = './data_snapshots'
    if epoch is None:
        csv_files = glob.glob(path + "/*.csv")
        df_list = (pd.read_csv(file) for file in csv_files)
        big_df = pd.concat(df_list, ignore_index=True)
        return big_df
    else:
        return pd.read_csv(path + f"/data_snapshot_{epoch}.csv")


def rank_suff(loc):
    if loc == 1:
        return ("st")
    elif loc == 2:
        return ("nd")
    elif loc == 3:
        return ("rd")
    else:
        return ("th")


def get_messages(bot_name, data):
    assert bot_name in ["NMABOT", "NMTBOT", "NMSBOT", "MABOT", "MTBOT", "MSBOT"]
    assert data is not None

    bot_data = {"MABOT": {"bot_type": "all", "markov": True},
                "MTBOT": {"bot_type": "tops", "markov": True},
                "MSBOT": {"bot_type": "self", "markov": True},
                "NMABOT": {"bot_type": "all", "markov": False},
                "NMTBOT": {"bot_type": "tops", "markov": False},
                "NMSBOT": {"bot_type": "self", "markov": False}}

    bot_type, markov = bot_data[bot_name]["bot_type"], bot_data[bot_name]["markov"]
    queries = data.iloc[0][["query1", "query2", "query3"]].values.tolist()
    epoch = int(max(data["round_no"]))
    if markov:
        data = data[data["round_no"] == epoch]

    messages = [
        {"role": "system",
         "content": fr"You are an expert on: " + ", ".join(queries)},
        {"role": "system",
         "content": "You participate in a text relevance to said topics contest, your document should be ranked first "
                    "(1st) for every topic respectively."},
        {"role": "user",
         "content": "Generate a single text that addresses the three topics in a comprehensive, informative and "
                    "coherent manner without any abrupt transitions or incomplete sentences.\nText:"},
    ]

    rounds = data['round_no'].unique()
    for r in rounds:
        round_data = data[data["round_no"] == r]
        top_user = get_top_user(round_data, r)
        top_ranks = round_data[round_data.username == top_user].iloc[0][
            ["position1", "position2", "position3"]].values.tolist()
        curr_text = round_data[round_data.username == bot_name].iloc[0]["posted_document"]
        curr_ranks = round_data[round_data.username == bot_name].iloc[0][
            ["position1", "position2", "position3"]].values.tolist()

        top_text = round_data[round_data.username == top_user].iloc[0]["posted_document"]

        messages.append({"role": "assistant", "content": f"{curr_text}"})
        messages.append({"role": "system",
                         "content": f"You were ranked {curr_ranks[0]}{rank_suff(curr_ranks[0])}, {curr_ranks[1]}{rank_suff(curr_ranks[1])}, "
                                    f"{curr_ranks[2]}{rank_suff(curr_ranks[2])} respectively in this epoch"})

        if bot_type == "all":
            txt_rnk = ""
            for _, row in round_data.iterrows():
                if row["username"] != bot_name:
                    txt_rnk += f"Ranked {row['position1']}{rank_suff(row['position1'])}, {row['position2']}{rank_suff(row['position2'])}, " \
                               f"{row['position3']}{rank_suff(row['position3'])} respectively:\n{row['posted_document']}\n\n"
            messages.append(
                {"role": "system",
                 "content": f"The ranked documents of your opponents in this epoch are as follows:\n {txt_rnk}"})
            messages.append({"role": "system",
                             "content": f"Infer from the documents and rankings how to align well with the ranker's"
                                        f" features."})
            messages.append({"role": "user",
                             "content": "Generate a single text that addresses the three topics in a comprehensive, informative and "
                                        "coherent manner without any abrupt transitions or incomplete sentences.\nText:"})

        elif bot_type == "tops":
            messages.append(
                {"role": "system",
                 "content": f"The top document, ranked {top_ranks[0]}{rank_suff(top_ranks[0])}, "
                            f"{top_ranks[1]}{rank_suff(top_ranks[1])}, {top_ranks[2]}{rank_suff(top_ranks[2])} "
                            f"respectively: {top_text}"})
            messages.append({"role": "system",
                             "content": f"Infer from the top document how to align well with the ranker's features."})
            messages.append({"role": "user",
                             "content": "Generate a single text that addresses the three topics in a comprehensive, informative and "
                                        "coherent manner without any abrupt transitions or incomplete sentences.\nText:"})

        elif bot_type == "self":
            messages.append({"role": "user",
                             "content": "Generate a single text that addresses the three topics in a comprehensive, informative and "
                                        "coherent manner without any abrupt transitions or incomplete sentences.\nText:"})
    return messages


def get_comp_text(messages):
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
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=max_tokens,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
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


if __name__ == '__main__':
    orig = pd.read_csv("bot_followup.csv")
    bot_followup = orig[orig['text'].isna()]
    data = get_data()
    data[["position1", "position2", "position3"]] = data[["position1", "position2", "position3"]].astype(int) / 10
    data[["position1", "position2", "position3"]] = data[["position1", "position2", "position3"]].astype(int)
    len_ = len(bot_followup)
    for idx, row in bot_followup.iterrows():
        bot_name = row["username"]
        group = row["group"]
        query_id = row["query_id"]
        print(f"Starting {idx + 1}/{len_}: {bot_name}, {group}, {query_id}")
        rel_data = data[(data['group'] == group) & (data['query_id'] == query_id)]
        rel_data = rel_data.loc[rel_data[["position1", "position2", "position3"]].median(axis=1).sort_values(0).index]
        messages = get_messages(bot_name, rel_data)
        res = get_comp_text(messages)['choices'][0]['message']['content']
        deleted_edition = min([x for x in [res.split(".")[-1], res.split("!")[-1], res.split("?")[-1], res.split("\n\n")[-1]] if x != res],
                              key=len)
        res = res.replace(deleted_edition, "")
        orig.at[idx, "text"] = res
        orig.to_csv("bot_followup.csv", index=False)
        print(f"Done {idx + 1}/{len_}: {bot_name}, {group}, {query_id}")
    x = 1
