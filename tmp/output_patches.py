#!/usr/bin/python

import os
import glob
import json
import pandas as pd

for output_path in glob.glob(os.path.join("outputs", "2025-06_*")):
    print(f"\nProcessing output path: {output_path}\n")

    with open(os.path.join(output_path, "description.json"), "r") as f:
        desc = json.load(f)
    
    ## 1. The file containing the ID of the participants in the training and test sets
    if desc["TestsetRatio"] != 0:
        split_ids_outpath = os.path.join(output_path, "train_and_test_ids.json")

        if not os.path.exists(split_ids_outpath):
            train_ids, test_ids = [], []

            for fp in glob.glob(os.path.join(output_path, "results_*_BEH.json")):
                with open(fp, "r") as f:
                    results = json.load(f)

                train_ids.append(results["TrainingSubjID"])
                test_ids.append(results["TestingSubjID"])

            split_with_ids = {
                "Train": sum(train_ids, []), # ensure a flat list
                "Test": sum(test_ids, [])
            }        
            with open(split_ids_outpath, 'w', encoding='utf-8') as f:
                json.dump(split_with_ids, f, ensure_ascii=False)

            print(f"Split IDs saved to: {split_ids_outpath}")

    ## 2. The file containing the features
    for fp in glob.glob(os.path.join(output_path, "features_*.csv")):
        df = pd.read_csv(fp, header=None)
        if df.iloc[0, 1] == 0:
            df = df.iloc[1:]
            df.to_csv(fp, index=False, header=False)

            print(f"Removed first row from: {fp}")

print("\nAll output paths processed :-)\n")