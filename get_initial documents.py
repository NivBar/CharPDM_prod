import json
import pickle
from tqdm import tqdm

import utils

with open('Archive/title_best_variations.pkl', 'rb') as handle:
    topic_dict = pickle.load(handle)

output = dict()

i = 1
start = 1
for idx, vals in tqdm(topic_dict.items()):
    if start > i:
        i += 1
        continue
    print("\n", vals["topic_title"], "\n")
    queries, description, backstory = [x.replace("\"", "'") for x in vals["queries"]], vals["description"].replace(
        "\"", "'"), vals["backstory"].replace("\"", "'")
    doc = utils.get_initial_doc(backstory, queries)
    txt_ = doc.choices[0].text
    output[str(idx)] = {"queries": queries, "description": description, "backstory": backstory, "doc": txt_}
    # print("\n", idx, output[str(idx)], "\n")
    with open(f'./pickles/{i}_docs.pickle', 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    i += 1

json_object = json.dumps(output, indent=4)

with open("topic_queries_doc.json", "w") as outfile:
    outfile.write(json_object)
