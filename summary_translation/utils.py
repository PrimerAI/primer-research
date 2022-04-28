import json
from collections import namedtuple

import numpy as np
import pandas as pd

# change data path here if needed
data_path = "data"

score_files = [
    "scores_jshannon.json",
    "scores_bleu.json",
    "scores_rouge.json",
    "scores_bertscores.json",
    "scores_blanc.json",
    "scores_estime.json",
]

qualities = ["coherence", "consistency", "fluency", "relevance"]

####################################################################################################


def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_json(path):
    with open(path) as f:
        return json.load(f)


def nested(d, name):
    data = []
    for language in d:
        for metric, values in d[language].items():
            if metric == "t_sec":
                continue
            else:
                data.append([name, language, metric] + values)

    return data


def single(d, name):
    data = []
    for language, values in d.items():
        data.append([name, language, None] + values)

    return data


def expert_proc(l):
    data = []
    for idx in range(4):
        data.append(["experts", "en^en", qualities[idx]] + l[idx])

    return data


def average_expert_scores(raw_scores):
    avg = []
    for quality in raw_scores:
        score_vectors = np.vstack([np.array(x) for x in quality])
        avg.append(score_vectors.mean(axis=0).tolist())

    return avg


def read_data_into_named_tuple():
    Scores = namedtuple(
        "Scores", ["experts", "js", "bleu", "rouge", "berts", "blanc", "estime"]
    )
    with open(f"{data_path}/scores_human_from_SummEval.json", "r") as f:
        experts = average_expert_scores(json.load(f))

    score_dicts = [experts]
    for file in score_files:
        with open(f"{data_path}/{file}", "r") as f:
            score_dicts.append(json.load(f))

    data = Scores(*score_dicts)

    return data


def read_data_into_dataframe():

    expert = average_expert_scores(
        read_json(f"{data_path}/scores_human_from_SummEval.json")
    )
    bertscores = read_json(f"{data_path}/scores_bertscores.json")
    bleu = read_json(f"{data_path}/scores_bleu.json")
    rouge = read_json(f"{data_path}/scores_rouge.json")
    jshannon = read_json(f"{data_path}/scores_jshannon.json")
    blanc = read_json(f"{data_path}/scores_blanc.json")
    estime = read_json(f"{data_path}/scores_estime.json")

    all_data = [
        nested(bertscores, "bertscore"),
        nested(rouge, "rouge"),
        expert_proc(expert),
        single(bleu, "bleu"),
        single(jshannon, "jshannon"),
        single(blanc, "blanc"),
        single(estime, "estime"),
    ]

    all_df = pd.concat(map(pd.DataFrame, all_data))
    all_df.columns = ["metric", "language", "submetric"] + list(all_df.columns)[3:]

    df = pd.melt(
        all_df,
        id_vars=["metric", "language", "submetric"],
        var_name="data_idx",
        value_name="value",
    )
    df["submetric"] = df.apply(
        lambda x: x.metric if x.submetric is None else x.submetric, axis=1
    )

    df.data_idx = df.data_idx - 3
    all_data = df.loc[df.language.str.startswith("en")].reset_index(drop=True).copy()
    all_data["from"] = all_data.language.apply(lambda x: x.split("^")[0])
    all_data["to"] = all_data.language.apply(lambda x: x.split("^")[-1])
    all_data["language"] = all_data.to

    return all_data
