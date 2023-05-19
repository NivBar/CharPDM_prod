import pandas as pd
from pprint import pprint

texts = pd.read_csv("bot_followup.csv").sort_values('query_id')
names = pd.read_csv("competition-topics.tsv", sep="\t").set_index('TRECTopicNumber').to_dict()['TRECQuery']
for idx, row in texts.iterrows():
    print(f"id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
    print(f"{row.text}\n\n")