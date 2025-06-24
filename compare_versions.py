#!/usr/bin/python

import os
import re
import sys
import json
import numpy as np
import pandas as pd
import itertools 
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
                if age_group == "all":
                    group = sex
                else:
                    group = {"le-44": "Y", "ge-45": "O"}[age_group] + sex
                data_path = os.path.join(input_folder, f"results_{age_group}_{sex}_{ori_name}.json")
            else:
                if label == "all":
                    group = label
                else:
                    group = {"le-44": "Y", "ge-45": "O"}[label]
                data_path = os.path.join(input_folder, f"results_{label}_{ori_name}.json")

            with open(data_path, 'r', errors='ignore') as f:
                results = json.load(f)
                selected_results = pd.DataFrame({ 
                    k: v for k, v in results.items() if k in ["Age", "PredictedAgeDifference", "CorrectedPAD"] 
                })
                selected_results["SID"] = [ x.replace("sub-0", "") for x in results[desc.sid_name] ]
                selected_results["Type"] = ori_name
                selected_results["Group"] = group
                selected_results["PAD"] = selected_results["PredictedAgeDifference"].abs()
                selected_results["PAD_ac"] = selected_results["CorrectedPAD"].abs()
                selected_result_list.append(selected_results)

    return pd.concat(selected_result_list, ignore_index=True)

def plot_categorical_bars(result_DF, pad_col, version_list, color_dict):
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.2)
    g = sns.catplot(
        data=result_DF, x="Group", y=pad_col, kind="bar", errorbar="se", sharex=False, 
        col="Version", col_order=version_list,
        hue="Version", hue_order=version_list, palette=color_dict, 
        height=5, aspect=.6, alpha=.8, legend=None
    )
    for col_val, ax in g.axes_dict.items():
        pad_mean = result_DF.query(f"Version == '{col_val}'")[pad_col].mean()
        ax.axhline(
            y=pad_mean, color=color_dict[col_val], linestyle='--'
        )
        x_pos = ax.get_xaxis().get_majorticklocs()
        ax.text(
            x=(x_pos[0] + x_pos[-1]) / 2, 
            y=ax.get_ylim()[1] + .1, 
            s=f"mean = {pad_mean:.2f}", 
            ha="center", va="bottom", color="k", fontsize=24
        )
        ax.set_title("")
        ax.tick_params(axis="both", which="major", labelsize=20)
    g.set_xlabels("")
    g.set_ylabels("")

    return g

def plot_bars_with_stats(result_DF, pad_col, version_list, stats_DF, color_dict, 
                         potential_y_lim=None, print_p=False):
    y_pos = max([stats_DF["V1_mean"].max(), stats_DF["V2_mean"].max()])
    y_offset = max([y_pos, potential_y_lim]) * .2
    p_offset = max([y_pos, potential_y_lim]) * .05
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.2)
    plt.figure(figsize=(6, 5))
    g = sns.barplot(
        data=result_DF, x="Version", y=pad_col, order=version_list, 
        hue="Version", hue_order=version_list, palette=color_dict, 
    )
    for x1, x2 in itertools.combinations(range(len(version_list)), 2):
        x_pos_1 = g.get_xaxis().get_majorticklocs()[x1]
        x_pos_2 = g.get_xaxis().get_majorticklocs()[x2]
        v1, v2 = version_list[x1], version_list[x2]
        stats_res = stats_DF.query(f"V1 == '{v1}' & V2 == '{v2}'")
        if len(stats_res) == 0:
            stats_res = stats_DF.query(f"V1 == '{v2}' & V2 == '{v1}'")
        p_val = stats_res.iloc[0]["P_value"]
        if p_val < .05:
            y_pos += y_offset
            g.plot(
                [x_pos_1, x_pos_2], [y_pos, y_pos], color="k", lw=1.5
            )
            p_sig = stats_res.iloc[0]["P_sig"]
            if print_p:
                p_sig = "< .001 ***" if p_val < .001 else f"{p_val:.3f} {p_sig}"
                y_pos += p_offset 
            else:
                y_pos -= p_offset
            g.text(
                (x_pos_1 + x_pos_2) / 2, y_pos, p_sig, 
                ha="center", va="bottom", fontsize=24, fontdict={"style": "italic"}
            )
    g.spines["right"].set_visible(False)
    g.spines["top"].set_visible(False)
    g.set_xlabel("")
    g.set_ylabel("")
    g.set_xticks(g.get_xaxis().get_majorticklocs())
    g.set_xticklabels([ f"#{i+1}" for i in range(len(version_list))], fontsize=20)
    g.tick_params(axis="y", which="major", labelsize=20)

    return g

def plot_color_legend(color_dict, fig_size, output_path, box_size=.3):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis("off")
    for idx, (label, color) in enumerate(color_dict.items()):
        rect = mpatches.Rectangle(
            (idx*3, 0), # 3 is the horizontal spacing unit
            width=box_size, height=box_size, color=color
        )
        ax.add_patch(rect)
        ax.text(
            x=(idx*3 + box_size + .2), # 0.2 units to the right of the color block
            y=(box_size / 2), # half the height of the color block
            s=f"[{idx+1}] {label}", ha='left', va='center', fontsize=12
        )
    ax.set_xlim(-.5, len(color_dict)*3) # start from -0.5 to make the first color block not touch the edge
    ax.set_ylim(-.1, box_size + .1) # leave some space at the top and bottom
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nColor legend is saved to:\n{output_path}")

## Main execution ---------------------------------------------------------------------

if __name__ == "__main__":
    note, input_folders, output_folder = {
        0: (
            "Original data and select features with ElasticNet.", 
            {
                version: os.path.join("outputs", folder) for version, folder in 
                dict(zip(
                    ["By_Age-Sex", "By_Age", "By_Sex", "Undivided"], 
                    [ f"2025-06_original_ElasticNetCV{suffix}" for suffix in ["", "_sex-0", "_age-0", "_age-0_sex-0"] ]
                )).items()
            }, 
            os.path.join("derivatives", "2025-06_original_ElasticNetCV_compare")
        ), 
        1: (
            "Original data and select features with RF-based permutation importance.", 
            {
                version: os.path.join("outputs", folder) for version, folder in 
                dict(zip(
                    ["By_Age-Sex", "By_Age", "By_Sex", "Undivided"], 
                    [ f"2025-06_original_RF-Permute{suffix}" for suffix in ["", "_sex-0", "_age-0", "_age-0_sex-0"] ]
                )).items()
            }, 
            os.path.join("derivatives", "2025-06_original_RF-Permute_compare")
        ), 
        2: (
            "Original data and select features with RF-based SHAP importance.", 
            {
                version: os.path.join("outputs", folder) for version, folder in 
                dict(zip(
                    ["By_Age-Sex", "By_Age", "By_Sex", "Undivided"], 
                    [ f"2025-06_original_RF-SHAP{suffix}" for suffix in ["", "_sex-0", "_age-0", "_age-0_sex-0"] ]
                )).items()
            }, 
            os.path.join("derivatives", "2025-06_original_RF-SHAP_compare")
        ), 
        3: (
            "Original data and select features with LGBM-based SHAP importance.", 
            {
                version: os.path.join("outputs", folder) for version, folder in 
                dict(zip(
                    ["By_Age-Sex", "By_Age", "By_Sex", "Undivided"], 
                    [ f"2025-06_original_LGBM-SHAP{suffix}" for suffix in ["", "_sex-0", "_age-0", "_age-0_sex-0"] ]
                )).items()
            }, 
            os.path.join("derivatives", "2025-06_original_LGBM-SHAP_compare")
        )
    }[int(sys.argv[1])]
    version_list = list(input_folders.keys())
    print(f"\n# Version comparison: {note}")

    ## Make output folder
    os.makedirs(output_folder, exist_ok=True)

    ## Save version notes
    notes = {"Note": note}
    notes.update(input_folders)
    with open(os.path.join(output_folder, "version notes.json"), 'w', encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False)

    ## Setup color legend (and save)
    color_dict = dict(zip(version_list, sns.color_palette("husl", len(version_list))))
    plot_color_legend(
        color_dict=color_dict, 
        fig_size=(len(version_list)*2, .5),
        output_path=os.path.join(output_folder, f"color_legend.png")
    )

    ## Load data
    result_DF_list = []
    for version, input_folder in input_folders.items():
        desc = load_description(input_folder)
        result_DF = load_data(input_folder, desc)
        result_DF["Version"] = version
        result_DF_list.append(result_DF)

    final_result_DF = pd.concat(result_DF_list, ignore_index=True)
    (
        final_result_DF
        .loc[:, ["Version", "Type", "SID", "Age", "Group", "PAD", "PAD_ac"]]
        .to_csv(os.path.join(output_folder, "concatenated_results.csv"), index=False)
    )

    for pad_name, pad_col in zip(["PAD", "PADAC"], ["PAD", "PAD_ac"]):

        ## Statistics:
        out_file = os.path.join(output_folder, f"compare_{pad_name}s.csv")

        stats_results = []
        for ori_name in desc.feature_orientations:

            for ver_1, ver_2 in itertools.combinations(input_folders.keys(), 2):
                V1_abs = final_result_DF.query(
                    f"Type == '{ori_name}' & Version == '{ver_1}'"
                )[pad_col]
                V2_abs = final_result_DF.query(
                    f"Type == '{ori_name}' & Version == '{ver_2}'"
                )[pad_col]

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
                # ## Paired sample t-test:
                # ttest_results = stats.ttest_rel(
                #     V1_abs, V2_abs, alternative="two-sided"
                # )
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
        print(f"\nStats results is saved to:\n{out_file}")

        ## Plots:
        SAME_SCALE = [True, False][int(sys.argv[2]) if len(sys.argv) > 2 else 0] 

        for ori_name in desc.feature_orientations:
            print(f"\nPlotting for {ori_name[:3]}...")

            g1_outpath = os.path.join(output_folder, f"compare_{ori_name[:3]}_{pad_name}s_bars.png")
            g1 = plot_categorical_bars(
                result_DF=final_result_DF.query(f"Type == '{ori_name}'"), 
                pad_col=pad_col, 
                version_list=version_list, 
                color_dict=color_dict
            )

            g2_outpath = os.path.join(output_folder, f"compare_{ori_name[:3]}_{pad_name}s_with_stats.png")
            g2 = plot_bars_with_stats(
                result_DF=final_result_DF.query(f"Type == '{ori_name}'"), 
                pad_col=pad_col, 
                version_list=version_list, 
                color_dict=color_dict, 
                stats_DF=stats_DF.query(f"Type == '{ori_name[:3]}'"), 
                potential_y_lim=g1.axes[0, 0].get_ylim()[1] if SAME_SCALE else 0
            )

            ## Ensure the same y-limits
            if SAME_SCALE:
                y_lim = max([
                    g1.axes[0, 0].get_ylim()[1], 
                    g2.get_ylim()[1]
                ])
            else:
                y_lim = None

            g1.set(ylim=(0, y_lim))
            g1.figure.tight_layout()
            g1.figure.savefig(g1_outpath)
            print(f"\nCategorical bar plot of {pad_col} is saved to:\n{g1_outpath}")
            plt.close(g1.figure)

            g2.set(ylim=(0, y_lim))
            g2.figure.tight_layout()
            g2.figure.savefig(g2_outpath)
            print(f"\nBar plot with stats of {pad_col} is saved to:\n{g2_outpath}")
            plt.close(g2.figure)
    
    print("\nDone!\n")

## Archive: ===========================================================================

# note, input_folders, output_folder = [
#     (
#         "Original data and select features with RF-based permutation importance.", 
#         {
#             version: os.path.join("outputs", folder) for version, folder in 
#             dict(zip(
#                 ["By_Age-Sex", "By_Age", "By_Sex", "Undivided"], 
#                 [ f"2025-06-17_original_{suffix}" for suffix in ["", "_sex-0", "_age-0", "_age-0_sex-0"] ]
#             ))
#         }, 
#         os.path.join("derivatives", "2025-06-17_original_compare")
#     )
#     (
#         "Original data and select features with PCA.", 
#         {
#             "By_Age-Sex": os.path.join("outputs", "2025-05-21_original_seed=9865"), 
#             "By_Age"    : os.path.join("outputs", "2025-05-23_original_seed=9865_sex-0"), 
#             "By_Sex"    : os.path.join("outputs", "2025-05-23_original_seed=9865_age-0"), 
#             "Undivided" : os.path.join("outputs", "2025-05-23_original_seed=9865_age-0_sex-0")
#         }, 
#         os.path.join("derivatives", "2025-05-23_original_seed=9865_compare")
#     ), 
#     (
#         "Down-sampled data and select top-50 features.", 
#         {
#             "By_Age-Sex": os.path.join("outputs", "2025-05-28_down-sampled_seed=9865"), 
#             "By_Age"    : os.path.join("outputs", "2025-05-28_down-sampled_seed=9865_sex-0"), 
#             "By_Sex"    : os.path.join("outputs", "2025-05-28_down-sampled_seed=9865_age-0"), 
#             "Undivided" : os.path.join("outputs", "2025-05-28_down-sampled_seed=9865_age-0_sex-0")
#         },
#         os.path.join("derivatives", "2025-05-28_down-sampled_seed=9865_compare")
#     ), 
#     (
#         "Original data and select top-50 features.", 
#         {
#             "By_Age-Sex": os.path.join("outputs", "2025-05-29_original_seed=9865"), 
#             "By_Age"    : os.path.join("outputs", "2025-05-29_original_seed=9865_sex-0"), 
#             "By_Sex"    : os.path.join("outputs", "2025-05-29_original_seed=9865_age-0"), 
#             "Undivided" : os.path.join("outputs", "2025-05-29_original_seed=9865_age-0_sex-0")
#         }, 
#         os.path.join("derivatives", "2025-05-29_original_seed=9865_compare")
#     ), 
#     (
#         "Down-sampled data and select features with PCA.", 
#         {
#             "By_Age-Sex": os.path.join("outputs", "2025-06-02_down-sampled_seed=9865"), 
#             "By_Age"    : os.path.join("outputs", "2025-06-02_down-sampled_seed=9865_sex-0"), 
#             "By_Sex"    : os.path.join("outputs", "2025-06-02_down-sampled_seed=9865_age-0"), 
#             "Undivided" : os.path.join("outputs", "2025-06-02_down-sampled_seed=9865_age-0_sex-0")
#         },
#         os.path.join("derivatives", "2025-06-02_down-sampled_seed=9865_compare")
#     )
# ]

