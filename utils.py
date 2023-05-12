import openai
import pandas as pd
import config


def get_initial_doc(description, subtopics):
    """
    creates the initial documents for the information retrieval competition.
    :param description: str, the information need
    :param subtopics: list of str, the subtopics of the information need
    :return:
    """
    max_tokens = config.max_tokens
    response = False
    prompt_ = f"Given a scenario depicting the information need, write a short text that the person described in the " \
              f"scenario would like to read. " \
              f"The text should cover the following subtopics: '{subtopics[0]}'," \
              f"'{subtopics[1]}', '{subtopics[2]}'\n\nScenario: {description}\n\nGenerated text:"

    while not response:
        try:
            response = openai.Completion.create(
                model=config.model,
                prompt=prompt_,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=max_tokens,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
            )
            # print("success")
            word_no = len(response.choices[0].text.split())
            if word_no > 150:
                max_tokens -= 50
                response = False
                print(f"word no: {word_no}, max tokens: {max_tokens}")
                continue
            break
        except Exception as e:
            print(e)
            continue
    return response


def data_set_creation():
    with open(".\content_modification_dataset\documents.trectext", "r", encoding="utf8") as f:
        xml = f.read().replace("&", "&amp;")
    xml = fr"<root>{xml}</root>"
    doc_df = pd.read_xml(xml).astype(str)

    qrel_df = \
        pd.read_csv(".\content_modification_dataset\documents.quality", header=None, delimiter=r"\s+").astype(
            str).rename(
            {2: "DOCNO", 3: "QREL"}, axis=1).replace(
            {'EPOCH': 'ROUND'}, regex=True)[["DOCNO", "QREL"]]

    ksrels_df = \
        pd.read_csv(".\content_modification_dataset\documents.relevance", header=None, delimiter=r"\s+").astype(
            str).rename({2: "DOCNO", 3: "KSREL"}, axis=1)[
            ["DOCNO", "KSREL"]]

    query_df = pd.read_csv(".\content_modification_dataset\queries.txt", header=None, delimiter=r":").astype(
        str).rename(
        {0: "query_id", 1: "query"},
        axis=1)

    pos_df = \
        pd.read_csv(".\content_modification_dataset\documents.positions", header=None, delimiter=r"\s+").astype(
            str).rename(
            {2: "DOCNO", 3: "POS"}, axis=1)[["DOCNO", "POS"]]

    merge_df = doc_df.merge(qrel_df, on="DOCNO").merge(ksrels_df, on="DOCNO").merge(pos_df, on="DOCNO")
    merge_df[["round_number", "query_id", "author_id"]] = merge_df["DOCNO"].apply(
        lambda x: pd.Series(str(x).split("-")[1:]))
    merge_df = merge_df.merge(query_df, on="query_id")[
        ['DOCNO', 'round_number', 'query_id', 'author_id', 'query', 'TEXT', 'QREL', 'KSREL', 'POS']]
    return merge_df


def gen_fine_tune_prompt(feature_text, query):
    fine_tune_prompt = f"Given the query '{query}', please write a superior feature text that better describes it " \
                       f"than the following inferior feature text: '{feature_text}'.\n\nGenerated text: "
    return fine_tune_prompt


def get_id_text_pairs(id_, epoch="last"):
    if epoch == 'last': epoch = max(list(config.comp_data.round_number))
    df = config.comp_data[(config.comp_data.query_id == id_) & (config.comp_data.round_number == epoch)][
        ["author_id", "TEXT"]]
    pairs = df.values.tolist()
    pair_dict = {p[0]: p[1] for p in pairs}
    return pair_dict
