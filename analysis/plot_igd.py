from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from utils import read_score_from_path

# -------------------------
# Step 1: True Pareto Front
# -------------------------
def calculate_true_pareto_front(folder_list: list[str]) -> np.ndarray:
    full_scores = []
    
    for folder in folder_list:
        folder_path = Path(folder)
        for file_path in folder_path.rglob("samples_1~200.json"):  # recursive search
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                scores = [item.get("score") for item in data if item.get("score") is not None]
                scores = [[abs(x) for x in pair] for pair in scores]
                full_scores.extend(scores)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    F_hist_np = np.array(full_scores)
    true_nd_indices = NonDominatedSorting().do(F_hist_np, only_non_dominated_front=True)
    true_pf_approx = F_hist_np[true_nd_indices]  # get true Pareto front
    
    return true_pf_approx

true_pf_approx = calculate_true_pareto_front(["logs/momcts", "logs/meoh"])
print(f"Length is: {len(true_pf_approx)}")

# -------------------------
# Step 2: IGD calculation
# -------------------------
def calculate_igd_from_path(json_path: str, true_pf_approx: np.ndarray) -> dict:
    F_hist = read_score_from_path(json_path)
    F_hist = np.array(F_hist)
    target_evals = sorted(set(range(10, len(F_hist), 10)) | {len(F_hist)})

    igd_at_targets = {}
    metric = IGD(true_pf_approx, zero_to_one=True)

    for target in target_evals:
        so_far = F_hist[:target+1]
        nd_idx = NonDominatedSorting().do(so_far, only_non_dominated_front=True) # get pareto front of heuristics at evaluation target
        P = so_far[nd_idx]
        igd_at_targets[target] = metric.do(P)

    return igd_at_targets

# -------------------------
# Step 3: Compare multiple files
# -------------------------
def compare_igd_curves(json_paths: list[str], labels: list[str]):
    plt.figure(figsize=(6, 4))

    for json_path, label in zip(json_paths, labels):
        igd_at_targets = calculate_igd_from_path(json_path, true_pf_approx)
        plt.plot(list(igd_at_targets.keys()), 
                 list(igd_at_targets.values()), 
                 marker="o", label=label)

    # plt.yscale("log")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.title("IGD Comparison")
    plt.show()

# -------------------------
# Run example
# -------------------------
if __name__ == "__main__":
    compare_igd_curves(
        [
            "logs/meoh/200_samples_v1/samples/samples_1~200.json",
            "logs/meoh/200_samples_v2/samples/samples_1~200.json",
            "logs/meoh/200_samples_v3/samples/samples_1~200.json",
            "logs/momcts/200_samples_v1/samples/samples_1~200.json",
            "logs/momcts/200_samples_v2/samples/samples_1~200.json", 
            "logs/momcts/200_samples_v3_lambda_0.1/samples/samples_1~200.json",
        ],
        labels=["MEOH", "MEOH_v2","MEOH_v3","MoMCTS_v1", "MoMCTS_v2", 'MoMCTS_v3']
    )
