import pandas as pd
from pprint import pprint

texts = pd.read_csv("bot_followup.csv").sort_values('query_id')
names = pd.read_csv("competition-topics.tsv", sep="\t").set_index('TRECTopicNumber').to_dict()['TRECQuery']
long_texts = []
short_texts = []
for idx, row in texts.iterrows():
    if len(row.text.split(' ')) < 140:
        print("SHORT TEXT\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
        short_texts.append(idx + 2)

    if len(row.text.split(' ')) > 150:
        print("LONG TEXT! CHANGE!\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
        long_texts.append(idx + 2)

    if "rank" in row.text:
        print("RANK IN TEXT\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
        print(f"{row.text}\n\n")
    if "LOOP BREAK" in row.text:
        print("LOOP BREAK IN TEXT\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")

    # print(f"idx in file: {idx+2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
    # print(f"{row.text}\n\n")
print("long texts:", sorted(long_texts))
print("short texts:", sorted(short_texts))