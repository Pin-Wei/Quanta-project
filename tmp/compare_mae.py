#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from itertools import product

selected_result_list = []

for age_group, sex in product(["le-44", "ge-45"], ["M", "F"]):
    group_name = f"{age_group}_{sex}"

    for ori_name in ["BEH"]:
        file_name = f"results_{group_name}_{ori_name}.json"

        for x, folder_name in enumerate([
            "2025-03-30_original", 
            "2025-05-02_original_seed=9865"
        ]):
            data_path = os.path.join("outputs", folder_name, file_name)
            with open(data_path, 'r', errors='ignore') as f:
                results = json.load(f)
            
            if x == 0:
                best_model = results["Model"]
                cache = best_model
            else:
                best_model = cache

            selected_results = pd.DataFrame({
                "Sex": sex, 
                "AgeGroup": age_group, 
                "Version": x, 
                "BestModel": best_model,  
                "MAE": round(results["MeanTrainMAE"], 3)
            }, index=[0])
            selected_result_list.append(selected_results)

long_DF = pd.concat(selected_result_list, ignore_index=True)

wide_DF = long_DF.pivot(
    index=["Sex", "AgeGroup", "BestModel"], 
    columns="Version", 
    values="MAE"
).rename(
    columns={
        0: "BestModel_MAE", 
        1: "ElasticNet_MAE"
    }
).reset_index()

wide_DF["MAE_diff"] = wide_DF["BestModel_MAE"] - wide_DF["ElasticNet_MAE"]

wide_DF.to_csv(
    os.path.join("derivatives", "MAE_comparison.csv"), 
    index=False
)
