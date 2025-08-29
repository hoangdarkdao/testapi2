import os
import json

folder_path = "./logs"

# Collect all matching JSON files
json_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(folder_path)
    for file in files
    if file.endswith("~200.json")
]

score_list = []

# Read scores from files
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        for entry in data:
            score = entry.get("score")
            if score is not None:  # skip None
                score_list.append(score)

# Initialize bounds (support arbitrary dimension)
if score_list:
    dim = len(score_list[0])
    max_bound = [float("-inf")] * dim
    min_bound = [float("inf")] * dim

    # Update bounds
    for score in score_list:
        if score is None:
            continue
        for i, val in enumerate(score):
            max_bound[i] = max(max_bound[i], val)
            min_bound[i] = min(min_bound[i], val)

    print("Max bound:", max_bound)
    print("Min bound:", min_bound)
else:
    print("⚠️ No valid scores found.")
