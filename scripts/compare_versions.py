#!/usr/bin/python

# python compare_versions.py <version_index> <same_scale>

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import itertools 
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

from gen_derivatives import (
    Constants, Description, load_model_results
)
sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from plotting import (
    plot_categorical_bars, plot_bars_with_stats, plot_color_legend
)

## Classes: ===========================================================================

class Config:
    def __init__(self, args):
        self.input_folders = {
            version: os.path.join("..", "outputs", folder) for version, folder in dict(zip(
                ["ByAgeSex", "ByAge", "BySex", "Undivided"], 
                [args.by_age_sex, args.by_age, args.by_sex, args.unstratified]
            )).items() if folder is not None
        }
        folder_name = f"{datetime.today().strftime('%Y-%m-%d')}_compare-versions"
        self.output_folder = os.path.join("..", "derivatives", folder_name)
        while os.path.exists(self.output_folder) and not args.overwrite:
            self.output_folder += "+"
        
        self.notes_outpath            = os.path.join(self.output_folder, "version_notes.json")
        self.color_legend_outpath     = os.path.join(self.output_folder, "color_legend.png")
        self.results_outpath          = os.path.join(self.output_folder, "results_DF_{}.csv")
        self.combined_results_outpath = os.path.join(self.output_folder, "combined_results_DF.csv")
        self.pad_pairstats_outpath    = os.path.join(self.output_folder, "compare_{}s.csv")
        self.pad_barplot_outpath      = os.path.join(self.output_folder, "[bar] compare_{}_{}s.png")
        self.pad_bar_stats_outpath    = os.path.join(self.output_folder, "[bar] compare_{}_{}s_with_stats.png")

## Functions: =========================================================================

def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ba", "--by_age", type=str, default=None, 
                        help="Output folder where models trained by age groups are stored.")
    parser.add_argument("-bs", "--by_sex", type=str, default=None, 
                        help="Output folder where models trained by gender are stored.")
    parser.add_argument("-bas", "--by_age_sex", type=str, default=None, 
                        help="Output folder where models trained by age groups and gender are stored.")
    parser.add_argument("-un", "--unstratified", type=str, default=None, 
                        help="Output folder where unstratified models are stored.")
    parser.add_argument("-ss", "--same_scale", action="store_true", default=False, 
                        help="Whether to use the same scale for all versions.")
    parser.add_argument("-dp", "--different_participants", action="store_true", default=False, 
                        help="Whether the versions contain different participants.")
    parser.add_argument("-n", "--note", type=str, default="", 
                        help="Note for the version comparison.")
    parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
                        help="Whether to overwrite existing files.")
    return parser.parse_args()

def compare_pad_values(V1_abs, V2_abs, independent=False):
    V1_mean, V2_mean = V1_abs.mean(), V2_abs.mean()
    V1_std, V2_std = V1_abs.std(), V2_abs.std()

    ## Levene's test for homogeneity of variance:
    levene_stats, levene_p = stats.levene(V1_abs, V2_abs)                
    if levene_p < 0.05:
        equal_var = False
    else:
        equal_var = True

    if independent: # Independent sample t-test
        ttest_results = stats.ttest_ind(
            V1_abs, V2_abs, equal_var=equal_var, alternative="two-sided"
        )
    else: # Paired sample t-test
        ttest_results = stats.ttest_rel(
            V1_abs, V2_abs, alternative="two-sided"
        )
    t_stat = ttest_results.statistic
    p_value = ttest_results.pvalue
    df = ttest_results.df

    ## Cohen's D:
    if len(V1_abs) == len(V2_abs): # same sample size
        s_pooled = np.sqrt((V1_std**2 + V2_std**2) / 2)
    else:
        s_pooled = np.sqrt(
            ((len(V1_abs)-1) * V1_std**2 + (len(V2_abs)-1) * V2_std**2) / 
            (len(V1_abs) + len(V2_abs) -2)
        )
    d = (V1_mean - V2_mean) / s_pooled

    return pd.DataFrame({
        "V1_mean": V1_mean, 
        "V2_mean": V2_mean, 
        "V1_std": V1_std,
        "V2_std": V2_std, 
        "Cohen_d": d, 
        "Levene_stat": levene_stats,
        "Levene_p": levene_p, 
        "Equal_var": str(equal_var)[:1], 
        "DF": df, 
        "T_stat": t_stat, 
        "P_value": p_value, 
        "P_sig": "*" * sum( p_value <= t for t in [0.05, 0.01, 0.001] )
    }, index=[0])

## Main function: ---------------------------------------------------------------------

def main():
    args = define_arguments()
    const = Constants()
    config = Config(args)

    ## Check if at least one input folder is provided
    if len(config.input_folders) < 2:
        raise ValueError("Please provide at least two input folder.")
    
    ## Create output folder
    os.makedirs(config.output_folder, exist_ok=True)
    print(f"\nOutput folder: {config.output_folder}")

    ## Save version notes
    notes = {
        "Note": args.note, 
        "Different_Participants": args.different_participants,
        "Plot_in_Same_Scale": args.same_scale
    }
    notes.update(config.input_folders)
    with open(config.notes_outpath, 'w', encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False)
    print(f"\nVersion notes is saved to:\n{config.notes_outpath}")

    ## Setup color legend (and save)
    version_list = list(config.input_folders.keys())
    color_dict = dict(zip(version_list, sns.color_palette("husl", len(version_list))))
    plot_color_legend(
        color_dict=color_dict, 
        fig_size=(len(version_list)*2, .5), 
        output_path=config.color_legend_outpath, 
        overwrite=args.overwrite
    )
    print()

    ## Load data
    result_DF_list, feature_ori_sets = [], []
    args.ignore_all = False # required for loading description

    for version, input_folder in config.input_folders.items():

        ## Load description
        desc_json_path = os.path.join(input_folder, "description.json")
        with open(desc_json_path, 'r', errors='ignore') as f:
            desc_json = json.load(f)
        desc = Description(desc_json, const, args)

        ## Load model results
        config.result_path = os.path.join(input_folder, "results_{}_{}.json")
        result_DF, _, _ = load_model_results(
            config, desc, const, 
            output_path=config.results_outpath.format(version), 
            overwrite=args.overwrite
        )

        result_DF["Version"] = version
        result_DF["PAD_abs"] = result_DF["PredictedAgeDifference"].abs()
        result_DF["PADAC_abs"] = result_DF["CorrectedPAD"].abs()
        if "Sex" not in result_DF.columns:
            result_DF["Sex"] = ""
            result_DF.replace({"AgeGroup": {"le-44": "Y", "ge-45": "O"}}, inplace=True)
        else:
            result_DF.replace({"AgeGroup": {"all": "", "le-44": "Y", "ge-45": "O"}}, inplace=True)
        
        result_DF["Group"] = result_DF["AgeGroup"] + result_DF["Sex"]

        result_DF_list.append(result_DF)
        print()

    final_result_DF = pd.concat(result_DF_list, ignore_index=True)
    cols = ["Version", "Type", "Group", "SID", "Age", "PAD_abs", "PADAC_abs"]
    final_result_DF.loc[:, cols].to_csv(config.combined_results_outpath, index=False)

    feature_orientations = final_result_DF["Type"].unique() # only the first 3 characters of each orientation

    ## Compare PAD and PADAC values between versions  
    for pad_col in ["PAD_abs", "PADAC_abs"]:

        ## Stats
        stats_DF_list = []
        for ori_name in feature_orientations:
            print(f"\nCalculating stats for {ori_name}...")

            for ver_1, ver_2 in itertools.combinations(version_list, 2):
                stats_results = compare_pad_values(
                    V1_abs=final_result_DF.query(f"Type == '{ori_name}' & Version == '{ver_1}'")[pad_col], 
                    V2_abs=final_result_DF.query(f"Type == '{ori_name}' & Version == '{ver_2}'")[pad_col], 
                    independent=True if args.different_participants else False
                )
                stats_results.insert(0, "Type", ori_name)
                stats_results.insert(1, "V1", ver_1)
                stats_results.insert(2, "V2", ver_2)
                stats_DF_list.append(stats_results)

        stats_DF = pd.concat(stats_DF_list, ignore_index=True)
        pad_pairstats_outpath = config.pad_pairstats_outpath.format(pad_col.replace("_abs", ""))
        stats_DF.to_csv(pad_pairstats_outpath, index=False)
        print(f"\nStats results is saved to:\n{pad_pairstats_outpath}")

        ## Plots:
        for ori_name in feature_orientations:
            print(f"\nPlotting for {ori_name}...")

            g1_fig_outpath = config.pad_barplot_outpath.format(ori_name, pad_col.replace("_abs", ""))
            g1 = plot_categorical_bars(
                DF=final_result_DF.query(f"Type == '{ori_name}'"), 
                pad_col=pad_col, 
                version_list=version_list, 
                color_dict=color_dict
            )

            g2_fig_outpath = config.pad_bar_stats_outpath.format(ori_name, pad_col.replace("_abs", ""))
            g2 = plot_bars_with_stats(
                result_DF=final_result_DF.query(f"Type == '{ori_name}'"), 
                stats_DF=stats_DF.query(f"Type == '{ori_name}'"),
                pad_col=pad_col, 
                version_list=version_list, 
                color_dict=color_dict, 
                potential_y_lim=g1.axes[0, 0].get_ylim()[1] if args.same_scale else 0
            )

            ## Ensure the same y-limits
            if args.same_scale:
                y_lim = max([
                    g1.axes[0, 0].get_ylim()[1], 
                    g2.get_ylim()[1]
                ])
            else:
                y_lim = None

            g1.set(ylim=(0, y_lim))
            g1.figure.tight_layout()
            g1.figure.savefig(g1_fig_outpath)
            print(f"\nCategorical bar plot of {pad_col} is saved to:\n{g1_fig_outpath}")
            plt.close(g1.figure)

            g2.set(ylim=(0, y_lim))
            g2.figure.tight_layout()
            g2.figure.savefig(g2_fig_outpath)
            print(f"\nBar plot with stats of {pad_col} is saved to:\n{g2_fig_outpath}")
            plt.close(g2.figure)

if __name__ == "__main__":
    main()
    print("\nDone!\n")