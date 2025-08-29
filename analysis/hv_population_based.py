import json
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume
from utils import read_score_from_path

def calculate_global_pareto_front_hv_multiple(json_paths, steps=10, visualize=True):
    """
    Args:
        json_paths: list of JSON file paths
        calculated hv based on population
    """
    plt.figure(figsize=(8, 5))
    
    for path in json_paths:
        # Read and prepare data
        F_hist = read_score_from_path(path)
        F_hist = np.array(F_hist, dtype=float)

        all_F = np.vstack(F_hist)
        approx_ideal = all_F.min(axis=0)
        approx_nadir = all_F.max(axis=0)

        metric = Hypervolume(
            ref_point=np.array([1.1, 1.1]),
            norm_ref_point=False,
            zero_to_one=True,
            ideal=approx_ideal,
            nadir=approx_nadir
        )

        n_evals = list(range(0, len(F_hist), steps))
        hv_values = [metric(F_hist[:i+1]) for i in n_evals]

        # Make label include algorithm name and version
        label_name = f"{path.split('/')[-4]}_{path.split('/')[-3]}"
        plt.plot(n_evals, hv_values, marker='o', linestyle='-', label=label_name)

    if visualize:
        plt.xlabel("Number of Evaluations")
        plt.ylabel("Hypervolume")
        plt.title("Hypervolume Progression Comparison")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    paths = [
        "logs/momcts/200_samples_v1/samples/samples_1~200.json",
        "logs/momcts/200_samples_v2/samples/samples_1~200.json",
        "logs/meoh/200_samples_v2/samples/samples_1~200.json",
        "logs/meoh/200_samples_v1/samples/samples_1~200.json",
        "logs/nsga2/200_samples_v1/samples/samples_1~200.json",
        "logs/nsga2/200_samples_v2/samples/samples_1~200.json",
        "logs/momcts/20250819_144144/samples/samples_1~200.json"
    ]
    calculate_global_pareto_front_hv_multiple(paths)
