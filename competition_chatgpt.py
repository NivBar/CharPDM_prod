import openai
import config
import tiktoken
import pandas as pd
import glob
import re
# from bson.objectid import ObjectId
import warnings
from pprint import pprint

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

def rank_str(loc):
    if loc == 1:
        return ("first")
    elif loc == 2:
        return ("second")
    elif loc == 3:
        return ("third")
    elif loc == 4:
        return ("fourth")
    elif loc == 5:
        return ("fifth")


def count_words_complete_sentences(text):
    # Split the text into sentences using a regex pattern
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)

    # Check if the last sentence is incomplete and remove it if necessary
    if sentences and sentences[-1].strip() and sentences[-1].strip()[-1] not in ['.', '!', '?']:
        sentences.pop()

    # Check if truncation is necessary
    if sentences:
        word_count = sum(len(sentence.split()) for sentence in sentences)
        truncated_text = text

        while word_count > 150:
            if len(sentences) < 2:
                break
            sentences.pop()
            truncated_text = ' '.join(sentences)
            word_count = sum(len(sentence.split()) for sentence in sentences)

        if word_count < 140:
            return len(text.split()), text, False

        if word_count >= 140 and word_count <= 150:
            return word_count, truncated_text + "." if truncated_text[-1] != "." else truncated_text, True

    # All sentences are complete
    word_count = len(text.split())
    return word_count, text, False

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
         "content": fr"You participate in a search engine optimization competition regarding the following queries: " +
                    ", ".join(queries) +
                    ". Your document should be ranked first for all queries."},
        {"role": "user",
         "content": "Generate a single text that a user will deem as highly relevant to all three queries."
                    " The text should be comprehensive, informative and coherent. Avoid meta commentary.\nText:"},
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
                         "content": f"You were ranked {rank_str(curr_ranks[0])}, {rank_str(curr_ranks[1])}, "
                                    f"{rank_str(curr_ranks[2])} respectively in this epoch"})

        if bot_type == "all":
            txt_rnk = ""
            for _, row in round_data.iterrows():
                if row["username"] != bot_name:
                    txt_rnk += f"Ranked {rank_str(row['position1'])}, {rank_str(row['position2'])}, " \
                               f"{rank_str(row['position3'])} respectively:\n{row['posted_document']}\n\n"
            messages.append(
                {"role": "system",
                 "content": f"The ranked documents of your opponents in this epoch are as follows:\n {txt_rnk}"})
            messages.append({"role": "system",
                             "content": f"Infer from the documents and rankings how to align well with the ranker's"
                                        f" features."})
            messages.append({"role": "user",
                             "content": "Generate a single text that a user will deem as highly relevant to all three queries."
                                        " The text should be comprehensive, informative and coherent. Avoid meta commentary.\nText:"})

        elif bot_type == "tops":
            messages.append(
                {"role": "system",
                 "content": f"The top document, ranked {rank_str(top_ranks[0])}, "
                            f"{rank_str(top_ranks[1])}, {rank_str(top_ranks[2])} "
                            f"respectively: {top_text}"})
            messages.append({"role": "system",
                             "content": f"Infer from the top document how to align well with the ranker's features."})
            messages.append({"role": "user",
                             "content": "Generate a single text that a user will deem as highly relevant to all three queries."
                                        " The text should be comprehensive, informative and coherent. Avoid meta commentary.\nText:"})

        elif bot_type == "self":
            messages.append({"role": "user",
                             "content": "Generate a single text that a user will deem as highly relevant to all three queries."
                                        " The text should be comprehensive, informative and coherent. Avoid meta commentary.\nText:"})
    return messages


def get_comp_text(messages,temperature = config.temperature,top_p = config.top_p, frequency_penalty = config.frequency_penalty,presence_penalty = config.presence_penalty):
    max_tokens = config.max_tokens
    response = False
    prompt_tokens = len(encoder.encode("".join([line['content'] for line in messages]))) + 200
    while prompt_tokens + max_tokens > 4096:
        max_tokens -= 50
        print("Changed max tokens for response to:", max_tokens)

    word_no, res = 0, ""

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
            word_no, res, ok_flag = count_words_complete_sentences(response['choices'][0]['message']['content'])
            if word_no > 150 or (word_no < 140 and max_tokens > 170):
                max_tokens -= 10
                response = False
                print(f"word no was: {word_no}, dropping max tokens to: {max_tokens}.")
                continue
            if word_no < 140 or not ok_flag:
                max_tokens += 10
                response = False
                print(f"word no was: {word_no}, increasing max tokens to: {max_tokens}.")
                continue
            break
        except Exception as e:
            print(e)
            continue
    print(f"word no is: {word_no}, current max tokes: {max_tokens}.")
    return res


if __name__ == '__main__':
    bot_valid = {"MABOT": False, "MTBOT": False, "MSBOT": False, "NMABOT": False, "NMTBOT": False, "NMSBOT": False}
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
        rel_data = rel_data.loc[
            rel_data[["position1", "position2", "position3"]].median(axis=1).sort_values(axis=0).index]
        messages = get_messages(bot_name, rel_data)

        if not bot_valid[bot_name]:
            pprint(messages)
            bot_valid[bot_name] = True

        res = get_comp_text(messages)
        # deleted_segment = min(
        #     [x for x in [res.split(".")[-1], res.split("!")[-1], res.split("?")[-1], res.split("\n\n")[-1]] if
        #      x != res],
        #     key=len)
        # res = res.replace(deleted_segment, "")
        orig.at[idx, "text"] = res
        orig.to_csv("bot_followup.csv", index=False)
        print(f"Done {idx + 1}/{len_}: {bot_name}, {group}, {query_id}")
    x = 1
