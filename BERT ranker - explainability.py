import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def infer_words_impact(texts, scores):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
    model = AutoModel.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

    # Tokenize the texts
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True)

    # Get the token IDs and offsets
    input_ids = tokenized_texts["input_ids"]
    offsets = tokenized_texts["offset_mapping"]

    # Get the embeddings for each text
    with torch.no_grad():
        outputs = model(input_ids)

    # Get the last hidden state
    last_hidden_state = outputs.last_hidden_state

    # Calculate the average embeddings for each text using offsets
    avg_embeddings = []
    for i in range(len(offsets)):
        start_offset = offsets[i][0][0]
        end_offset = offsets[i][-1][1]
        text_embedding = last_hidden_state[i, start_offset:end_offset, :].mean(dim=0).numpy()
        avg_embeddings.append(text_embedding)

    # Convert scores to numpy array
    scores = np.array(scores).reshape(-1, 1)

    # Concatenate embeddings and scores
    X = np.concatenate((avg_embeddings, scores), axis=1)

    # Find the words with the lowest scores
    lowest_score_words = []
    for i, text in enumerate(texts):
        words = tokenizer.tokenize(text)
        word_scores = X[i, :-1]
        lowest_score_word = words[np.argmin(word_scores)]
        lowest_score_words.append(lowest_score_word)

    return lowest_score_words

df = pd.read_csv("./data_snapshots/db_snapshot_6.csv").sample(frac=1).head(3)
df["avg_position"] = df.apply(lambda row: np.mean([row["position1"], row["position2"], row["position3"]])/10, axis=1)
texts = df["current_document"].tolist()
scores = df["avg_position"].tolist()

lowest_score_words = infer_words_impact(texts, scores)
for text, word in zip(texts, lowest_score_words):
    print(f"The word in '{text}' that leads to a lower score is '{word}'")
