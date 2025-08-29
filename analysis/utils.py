import json
import numpy as np
import os
import re
import glob
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def read_score_from_path(json_data: str) -> list[list[float, float]]:
   
    with open(json_data, "r") as f:
        data = json.load(f)
    scores = []
    for item in data:
        if 'score' in item and isinstance(item['score'], list) and len(item['score']) == 2:
            negated_score = [-s for s in item['score']]
            scores.append(negated_score)
        else:
            print(f"Warning: Skipping item due to invalid 'score' format: {item}")
    return scores

def find_pareto_front_from_scores(scores: list[list[float, float]]):
    F_hist_np = np.array(scores)
    nd_indices = NonDominatedSorting().do(F_hist_np, only_non_dominated_front=True)
    true_pf_approx = F_hist_np[nd_indices]

    return true_pf_approx


def read_population_scores_from_folder(folder_path: str) -> list[list[float, float]]:
    '''
    Args:
        mark = 0: the score is negative
        mark = 1: objective is positive
    '''
    mark = 0
    files = glob.glob(os.path.join(folder_path, "pop_*.json"))
    if len(files) == 0:
        mark = 1
        files = glob.glob(os.path.join(folder_path, "population_generation_*.json"))
        
    files.sort(key=lambda x: int(re.search(r"(\d+)", os.path.basename(x)).group()))
    data_list = []

    for file_path in files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = None  # or {}
            data_list.append({
                "filename": os.path.basename(file_path),
                "content": data
            })

    F_list = []
    for data in data_list:
        F = []
        for x in data["content"]:
            if mark == 0:
                obj, runtime = x["score"]
                obj, runtime = -obj, -runtime
            else:
                obj, runtime = x["score"]
                obj, runtime = -obj, -runtime
            F.append([obj, runtime])
        F_list.append(F)

    return F_list