import pandas as pd
import requests
import os
from pprint import pprint
import numpy as np

# TODO: change epoch
epoch = 3


##### create test texts #####

def get_sample_texts(df, n, group, epoch):
    df = df[df.group == group].sample(n=n)
    names = []
    for idx, row in df.iterrows():
        with open(f"./test_texts/{group}_{row.query_id}_{row.username}_{epoch}.txt", "w") as f:
            f.write(row.current_document)
            names.append(f"{group}_{row.query_id}_{row.username}_{epoch}")
    csv_df = pd.DataFrame(names, columns=["names"])
    csv_df["validation"] = np.nan
    csv_df.to_csv(f"./test_texts/{group}_{epoch}.csv", index=False)


df = pd.read_csv(f"./data_snapshots/db_snapshot_{epoch}.csv")
df = df[(df.group.isin(["B", "D"])) & ~(df.username.str.contains("BOT")) & (df.username != "dummy")]

get_sample_texts(df, 12, "B", epoch)
get_sample_texts(df, 12, "D", epoch)
x = 1
##### api test texts #####


def make_api_request(text):
    url = "https://api.gptzero.me/v2/predict/text"
    headers = {"Content-Type": "application/json"}
    data = {
        "document": text
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None


directory_path = "./test_texts"
files = []
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, "r") as file:
            file_content = file.read()
            data_dict = make_api_request(file_content)
            data_dict = data_dict["documents"]
            data_dict["name"] = filename
            files.append(data_dict)

# Example usage
input_text = "This is a test input."
api_response = make_api_request(input_text)

if api_response:
    print("API response:")
    pprint(api_response)
