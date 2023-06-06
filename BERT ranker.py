from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from competition_chatgpt import get_comp_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

def get_bert_scores(queries_to_docnos, docs_dict, queries_dict):
    scores_dict = {}
    for qid in tqdm(queries_to_docnos, desc="BERT", total=len(queries_to_docnos)):
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

def create_data_structures(df):
    queries_dict = dict(zip(df.query_id, df.query1))
    docs_dict = dict(zip(df.docno, df.current_document))
    queries_to_docnos = {k: list(set(df[df.query_id == k].docno)) for k in set(df.query_id)}
    return queries_dict, docs_dict, queries_to_docnos

all_files = glob.glob(os.path.join("./data_snapshots/", "*.csv"))
df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

# queries_dict, docs_dict, queries_to_docnos = create_data_structures(df[df.round_no == 5].drop_duplicates(subset=['current_document']))
# scores = get_bert_scores(queries_to_docnos, docs_dict, queries_dict)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate a batch of queries
queries = list(set(df.query1))

# Define the initial weights
initial_weights = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)

# Function to generate text using the weights
def generate_text(weights):
    # Implement your logic to generate text using the given weights
    # pass
    get_comp_text
    return "test text"

# Function to calculate similarity score between a query and generated text
def calculate_similarity(query, text):
    # Implement your logic to calculate similarity score
    pass

# Create the neural network model
model = Net()

# Use an optimizer to minimize the negative average score
optimizer = optim.Adam(model.parameters())

# Random train-test split
random.seed(42)
random.shuffle(queries)
train_queries = queries[:20]
val_queries = queries[20:]

# Lists to store losses
train_losses = []
val_losses = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Generate texts using the initial weights for train queries
    train_texts = [generate_text(initial_weights) for _ in range(len(train_queries))]

    # Calculate similarity scores for train queries
    train_scores = [calculate_similarity(q, t) for q, t in zip(train_queries, train_texts)]

    # Calculate negative average score as the train loss
    train_loss = -torch.mean(torch.tensor(train_scores))
    train_losses.append(train_loss.item())

    # Generate texts using the initial weights for validation queries
    val_texts = [generate_text(initial_weights) for _ in range(len(val_queries))]

    # Calculate similarity scores for validation queries
    val_scores = [calculate_similarity(q, t) for q, t in zip(val_queries, val_texts)]

    # Calculate negative average score as the validation loss
    val_loss = -torch.mean(torch.tensor(val_scores))
    val_losses.append(val_loss.item())

    # Backpropagation and weight update for train loss
    train_loss.backward()
    optimizer.step()

    # Update the initial weights for validation loss
    with torch.no_grad():
        initial_weights -= optimizer.param_groups[0]['lr'] * initial_weights.grad
        initial_weights.grad.zero_()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

# Get the optimized weights
optimized_weights = initial_weights

# Plotting the losses
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Optimized Weights:", optimized_weights)