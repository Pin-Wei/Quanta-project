#!/usr/bin/python

import os
import re
import sys
import json
import numpy as np
import pandas as pd
import itertools 
from scipy import stats

## Define functions -------------------------------------------------------------------

def load_description(input_folder):
    with open(os.path.join(input_folder, "description.json"), 'r') as f:
        desc_json = json.load(f)

    class Description:
        def __init__(self):
            for k, v in desc_json.items():
                snake_k = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", k).lower() # add underscores before capital letters that follow lowercase letters
                setattr(self, snake_k, v)

            self.sep_sex = bool(self.sex_separated) # rename to sep_sex for consistency
            self.age_breaks = [ 0 ] + [ int(x.split("-")[1]) for x in self.age_groups[:-1] ] + [ np.inf ]
            self.label_list = list(itertools.product(self.age_groups, ["M", "F"])) if self.sep_sex else self.age_groups
            self.sid_name = "TestingSubjID" if self.testset_ratio != 0 else "SubjID"

    return Description()

def load_data(input_folder, desc):
    selected_result_list = []
    for ori_name in desc.feature_orientations:
        for label in desc.label_list:
            if desc.sep_sex:
                age_group, sex = label
                data_path = os.path.join(input_folder, f"results_{age_group}_{sex}_{ori_name}.json")
            else:
                age_group = label
                data_path = os.path.join(input_folder, f"results_{age_group}_{ori_name}.json")

            with open(data_path, 'r', errors='ignore') as f:
                results = json.load(f)
                selected_results = pd.DataFrame({ 
                    k: v for k, v in results.items() if k in ["Age", "PredictedAgeDifference", "CorrectedPAD"] 
                })
                selected_results["SID"] = [ x.replace("sub-0", "") for x in results[desc.sid_name] ]
                selected_results["Type"] = ori_name
                selected_results["AgeGroup"] = age_group
                selected_results["Sex"] = sex if desc.sep_sex else None
                selected_result_list.append(selected_results)

    return pd.concat(selected_result_list, ignore_index=True)

## Main execution ---------------------------------------------------------------------

if __name__ == "__main__":
    note, input_folders, output_folder = [
        (
            "Original data and select features with PCA.", 
            {
                "By_Age-Sex": os.path.join("outputs", "2025-05-21_original_seed=9865"), 
                "By_Age"    : os.path.join("outputs", "2025-05-23_original_seed=9865_sex-0"), 
                "By_Sex"    : os.path.join("outputs", "2025-05-23_original_seed=9865_age-0"), 
                "Undivided" : os.path.join("outputs", "2025-05-23_original_seed=9865_age-0_sex-0")
            }, 
            os.path.join("derivatives", "2025-05-23_original_seed=9865_compare")
        ), 
        (
            "Down-sampled data and select top-50 features.", 
            {
                "By_Age-Sex": os.path.join("outputs", "2025-05-28_down-sampled_seed=9865"), 
                "By_Age"    : os.path.join("outputs", "2025-05-28_down-sampled_seed=9865_sex-0"), 
                "By_Sex"    : os.path.join("outputs", "2025-05-28_down-sampled_seed=9865_age-0"), 
                "Undivided" : os.path.join("outputs", "2025-05-28_down-sampled_seed=9865_age-0_sex-0")
            },
            os.path.join("derivatives", "2025-05-28_down-sampled_seed=9865_compare")
        )
    ][int(sys.argv[1])]
    print(f"\n# Version comparison: {note}\n")

    os.makedirs(output_folder, exist_ok=True)

    notes = {"Note": note}
    notes.update(input_folders)
    with open(os.path.join(output_folder, "version notes.json"), 'w', encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False)

    result_DF_list = []
    for version, input_folder in input_folders.items():
        desc = load_description(input_folder)
        result_DF = load_data(input_folder, desc)
        result_DF["Version"] = version
        result_DF_list.append(result_DF)

    final_result_DF = pd.concat(result_DF_list, ignore_index=True)

    for pad_name, pad_col in zip(
        ["PAD", "PADAC"], 
        ["PredictedAgeDifference", "CorrectedPAD"]
    ):
        out_file = os.path.join(output_folder, f"compare_{pad_name}s.csv")

        stats_results = []
        for ori_name in desc.feature_orientations:
            for ver_1, ver_2 in itertools.combinations(input_folders.keys(), 2):

                V1_abs = final_result_DF.query(
                    f"Type == '{ori_name}' & Version == '{ver_1}'"
                )[pad_col].abs()
                V2_abs = final_result_DF.query(
                    f"Type == '{ori_name}' & Version == '{ver_2}'"
                )[pad_col].abs()

                ## Levene's test for homogeneity of variance:
                levene_stats, levene_p = stats.levene(V1_abs, V2_abs)                
                if levene_p < 0.05:
                    equal_var = False
                else:
                    equal_var = True

                ## Independent sample t-test:
                ttest_results = stats.ttest_ind(
                    V1_abs, V2_abs, equal_var=equal_var, alternative="two-sided"
                )
                t_stat = ttest_results.statistic
                p_value = ttest_results.pvalue
                df = ttest_results.df

                if p_value < 0.001:
                    p_sig = "***"
                elif p_value < 0.01:
                    p_sig = "**"
                elif p_value < 0.05:
                    p_sig = "*"
                elif p_value < 0.1:
                    p_sig = "."
                else:
                    p_sig = ""
                    
                stats_results.append(
                    pd.DataFrame({
                        "Type": ori_name[:3], 
                        "V1": ver_1, 
                        "V2": ver_2, 
                        "V1_mean": V1_abs.mean(), 
                        "V2_mean": V2_abs.mean(), 
                        "V1_std": V1_abs.std(),
                        "V2_std": V2_abs.std(), 
                        "Levene_stat": levene_stats,
                        "Levene_p": levene_p, 
                        "Equal_var": str(equal_var)[:1], 
                        "DF": df, 
                        "T_stat": t_stat, 
                        "P_value": p_value, 
                        "P_sig": p_sig
                    }, index=[0])
                )

        stats_DF = pd.concat(stats_results, ignore_index=True)
        stats_DF.to_csv(out_file, index=False)
        print(f"Saved: {out_file}\n")

    print("Done!\n")