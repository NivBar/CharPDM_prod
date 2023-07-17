import json

import openai
import pandas as pd
from API_key import API_key


#### bot names ####
def get_names_dict(markov=False):
    if markov:
        return {"all": "MABOT", "tops": "MTBOT", "self": "MSBOT"}
    else:
        return {"all": "NMABOT", "tops": "NMTBOT", "self": "NMSBOT"}


#### visualization parameters ####
data_exist = {"comp": True, "improvements": True, "tops": True}
display_graphs = False

#### openai parameters ####
openai.api_key = API_key

"""
Model: Determines the architecture and parameters of the language model used for text generation. Different models have 
different strengths and weaknesses for specific types of text generation tasks.

Temperature: Controls the level of randomness and creativity in the generated text. High temperature values (e.g., 1.0 
or higher) can produce more diverse and unexpected outputs, while low values (e.g., 0.5 or lower) can produce more 
predictable and conservative outputs.

Top_p: Limits the set of possible next words based on the model's predictions. High top_p values (e.g., 0.9 or higher) 
allow for more variation and creativity, while low values (e.g., 0.1 or lower) can produce more predictable and 
conservative outputs.

Max_tokens: Sets an upper limit on the number of tokens that can be generated in the output text. High max_tokens values 
(e.g., 500 or higher) can produce longer outputs, while low values (e.g., 50 or lower) can produce shorter and more 
concise outputs.

Frequency_penalty: Encourages the model to generate less frequent words or phrases. High frequency_penalty values 
(e.g., 2.0 or higher) can increase the diversity and creativity of the generated text, while low values 
(e.g., 0.5 or lower) can produce more common and predictable outputs.

Presence_penalty: Encourages the model to avoid repeating words or phrases that have already appeared in the output 
text. High presence_penalty values (e.g., 2.0 or higher) can promote the generation of novel and varied text, while low 
values (e.g., 0.5 or lower) can produce more repetitive and redundant outputs.
"""
model = "gpt-3.5-turbo"
# temperature = 0.1
# top_p = 0.3
# max_tokens = 250
# frequency_penalty = 2.0
# presence_penalty = 0.0

temperature = 0.2
top_p = 0.3
max_tokens = 250
frequency_penalty = 1.0
presence_penalty = 0.0

#### useful data collections ####
# topic_codex_new = json.load(open("topic_queries_doc.json", "r"))
topic_codex = dict()

# TODO: change to actual copetition data when starting
comp_data = pd.read_csv("Archive/comp_dataset.csv")

query_index = {x[0]: x[1] for x in comp_data[["query_id", "query"]].drop_duplicates().values.tolist()}

