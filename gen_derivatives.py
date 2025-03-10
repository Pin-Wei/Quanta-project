#!/usr/bin/python

import os
import json
import argparse
import numpy as np
import pandas as pd
import itertools
import pingouin as pg
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

## Argument parser: -------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="""
    The dataset was divided into groups according to the gender and age range of the 
    participants, and different models were then trained based on the data collected 
    from three orientations (structural, behavioral, and functional) in these groups.
    
    This script generates the derivatives of the model outputs from the folder 
    specified by the '-f' argument, including:
                                 
    1) A table listing the model types and feature numbers used: 
        '[table] model types and feature numbers.csv'.
    2) A table listing the medians and standard deviations of ages for each participant group:
        '[table] data median & std.csv'.
    2+) If down-sampling (-ds) or bootstrapping (-bs) is specified, a folder named 
      '{sampling_method} (N={args.sample_size}; seed={seed})' will be created, 
      and a similar table will be generated after down-sampling or bootstrapping: 
        '[table] balanced data median & std.csv'.
    3) A bar plot comparing PAD values for different models:
        '[barplot] PAD values.png'.
    4) Matrices of pairwise correlations between PAD values for models trainied on 
      data collected from different orientations in each subject group: 
        '[cormat] {group_name}.png'.
    5) A table of pairwise correlations between PAD values for models trainied on 
      data collected from different orientations in each subject group: 
        '[table] pairwise corr.xlsx'.
    6) Scatter plots comparing PAD values for models trainied on data collected from
      different orientations in each subject group: 
        '[scatter] {group_name} ({ori_1} × {ori_2}).png'.

    The files will be saved in a folder with the same name as the '-f' argument
    under the 'derivatives' folder.
    """
)
parser.add_argument("-f", "--folder", type=str, required=True, 
                    help="The folder name of the model outputs.")
parser.add_argument("-ds", "--downsample", action="store_true", default=False, 
                    help="Down-sample the data without replacement.")
parser.add_argument("-bs", "--bootstrap", action="store_true", default=False, 
                    help="Down-sample the data with replacement (bootstrapping).")
parser.add_argument("-n", "--sample_size", type=int, default=50, 
                    help="The number of participants to down-sample to.")
parser.add_argument("-s", "--seed", type=int, default=None, 
                    help="The random seed for downsampling or bootstrapping.")
parser.add_argument("-pad", "--use_pad", action="store_true", default=False,
                    help="Use PAD values that have not been corrected for age.")
parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
                    help="Overwrite if the output files already exist.")
parser.add_argument("-pa", "--p_adjust", action="store_true", default=False,
                    help="Use adjusted p-values for pairwise correlations.")
args = parser.parse_args()

## Configuration class: ---------------------------------------------------------------

class Config:
    def __init__(self):
        self.folder = args.folder
        self.input_folder = os.path.join("outputs", self.folder)
        self.output_folder = os.path.join("derivatives", self.folder)
        self.model_info_filename    = "[table] model types and feature numbers.csv"
        self.disc_table_filename    = "[table] data median & std.csv"
        self.balanced_disc_table_fn = "[table] balanced data median & std.csv"
        self.barplot_filename       = "[barplot] PAD values.png"
        self.cormat_fn_template     = "[cormat] GroupName.png"
        self.corr_table_filename    = "[table] pairwise corr.xlsx"
        self.scatter_fn_template    = "[scatter] GroupName (Type1 × Type2).png"

## Load parameters: -------------------------------------------------------------------

def load_parameters(config):
    '''
    Load the parameters from the description file and update the configuration.
    <returns>:
    - config: Updated configuration
    '''
    ## Load description file: 
    desc_path = os.path.join(config.input_folder, "description.json")
    with open(desc_path, 'r', errors='ignore') as f:
        desc = json.load(f)

    ## Data groups:
    config.sep_sex = bool(desc["SexSeparated"]) 
    config.age_group_labels = desc["AgeGroups"]
    config.age_breaks = (
        [ 0 ] + 
        [ int(x.split("-")[0]) for x in config.age_group_labels[1:-1] ] + 
        [ int(config.age_group_labels[-1].split("-")[-1]) ] + 
        [ np.inf ]
    )
    config.label_cols = ["AgeGroup", "Sex"] if config.sep_sex else ["AgeGroup"]
    if config.sep_sex:
        config.label_list = list(itertools.product(config.age_group_labels, ["M", "F"]))
    else:
        config.label_list = config.age_group_labels   

    ## Used feature orientations (Options: ["STRUCTURE", "BEH", "FUNCTIONAL", "ALL"]):
    config.feature_orientations = desc["FeatureOrientations"] 

    ## If testset ratio is 0, then the data was not split into training and testing sets:
    config.sid_name = "SubjID" if int(desc["TestsetRatio"]) == 0 else "TestingSubjID"

    return config

## Load data: -------------------------------------------------------------------------

def load_data(config, selected_cols):
    '''
    Load the data from the input folder and return a DataFrame.
    <returns>:
    - DF: DataFrame with selected columns.
    '''
    selected_result_list = []
    print("\nLoading data...")

    for label in config.label_list:
        if config.sep_sex:
            age_group, sex = label
            group_name = f"{age_group}_{sex}"
        else:
            age_group = label
            group_name = label

        for ori_name in config.feature_orientations:
            data_path = os.path.join(
                config.input_folder, f"results_{group_name}_{ori_name}.json")
            
            if os.path.exists(data_path):
                print(os.path.basename(data_path))

                with open(data_path, 'r', errors='ignore') as f:
                    data = json.load(f)
                    selected_results = pd.DataFrame({ 
                        k: v for k, v in data.items() if k in selected_cols
                    })
                    selected_results["AgeGroup"] = age_group
                    if config.sep_sex:
                        selected_results["Sex"] = sex
                    selected_results["Type"] = ori_name
                    selected_results["Info"] = f"{data['Model']}_{data['NumberOfFeatures']}"
                    
                    selected_result_list.append(selected_results)

    DF = pd.concat(selected_result_list, ignore_index=True)

    return DF

## Function: --------------------------------------------------------------------------

def save_model_info(DF, label_cols, output_path, overwrite=False):
    '''
    Save the model types and feature numbers to a .csv table.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        model_info = (DF
            .loc[:, label_cols + ["Type", "Info"]]
            .drop_duplicates()
            .pivot(index = label_cols, 
                columns = "Type", 
                values = "Info")
        )
        model_info.to_csv(output_path)
        print(f"\nModel types and feature numbers are saved to:\n{output_path}")

## Function: --------------------------------------------------------------------------

def save_discriptive_table(DF, output_path):
    '''
    Save the median and standard deviation of the data to a .csv table.
    <returns>:
    - part_DF: DataFrame with unique subjects.
    - stats_table: Table with median and standard deviation of the data.
    '''
    part_DF = (DF
        .loc[:, ["SubjID", "Age", "Sex", "AgeGroup"]]
        .drop_duplicates("SubjID")
    )
    stats_table = (part_DF
        .groupby(["Sex", "AgeGroup"])["Age"]
        .agg(["count", "median", "std"])
        .reset_index()
    )
    stats_table.to_csv(output_path, index=False)
    print(f"\nData median and std are saved to:\n{output_path}")

    return part_DF, stats_table

## Function: --------------------------------------------------------------------------

def make_balanced_dataframe(part_DF, stats_table, seed, output_path, 
    iters_n=1000, med_diff_good=0.1, std_diff_good=0.5, alpha=0.5):
    '''
    Down-sample or bootstrap the data to make a balanced DataFrame
    where the median and STD of all sub-datas are similar, 
    and save the statistics to a .csv table.
    <returns>:
    - balanced_part_DF: Balanced DataFrame.
    '''
    target_med = stats_table.groupby("AgeGroup")["median"].mean()
    target_std = stats_table["std"].mean()
    sampled_df_list = []

    for (sex, age_group), group_data in part_DF.groupby(["Sex", "AgeGroup"]):
        best_sample, best_score = None, np.inf
        weights = np.ones(len(group_data))
    
        for _ in range(iters_n):
            current_is_good = 0
            if weights.max() != weights.min(): # normalize weights
                weights = (
                    (weights - weights.min()) / 
                    (weights.max() - weights.min())
                )
            sampled_df = group_data.sample(
                n=args.sample_size, 
                replace=args.bootstrap,  
                random_state=seed, 
                weights=weights / weights.sum()
            )
    
            ## Adjust weights in accordance to the current median:
            current_med = np.median(sampled_df["Age"].values)
            med_diff = current_med - target_med[age_group]

            if abs(med_diff) < med_diff_good:
                current_is_good += 1
            elif med_diff > 0: # increase the probability of sampling in lower-value areas if current median is too high
                weights[group_data["Age"] <= target_med[age_group]] *= 1 + alpha
            else: 
                weights[group_data["Age"] >= target_med[age_group]] *= 1 + alpha
            
            ## Adjust weights in accordance to the current standard deviation:
            current_std = np.std(sampled_df["Age"].values)
            std_diff = current_std - target_std

            if abs(std_diff) < std_diff_good:
                current_is_good += 1
            elif std_diff > 0: # decrease weight in proportion to distance if current std is too large
                weights -= alpha * abs(group_data["Age"].values - target_med[age_group]) 
            else: 
                weights += alpha * abs(group_data["Age"].values - target_med[age_group])
            
            ## Choose the best sample based on the score:
            score = (
                (abs(med_diff) / target_med[age_group]) + 
                (abs(std_diff) / target_std)
            )
            if current_is_good == 2: # stop if the current sample is good enough
                best_sample = sampled_df
                break
            elif score < best_score:
                best_sample = sampled_df
                best_score = score
    
        sampled_df_list.append(best_sample)
    
    balanced_part_DF = pd.concat(
        sampled_df_list, ignore_index=True
    )
    balanced_stats_table = (
        balanced_part_DF
        .groupby(["Sex", "AgeGroup"])["Age"]
        .agg(["count", "median", "std"])
        .reset_index()
    )
    balanced_stats_table.to_csv(output_path, index=False)
    print(f"\nThe median and std of balanced data are saved to:\n{output_path}")

    return balanced_part_DF

## Function: --------------------------------------------------------------------------

def modify_DF(used_DF, config):
    '''
    Modify the DataFrame and transform it to long format.
    <returns>:
    - DF_long: DataFrame in long format.
    '''
    used_DF["SID"] = used_DF["SubjID"].map(lambda x: x.replace("sub-0", ""))
    used_DF["PAD"] = np.abs(used_DF["PredictedAgeDifference"])
    used_DF["PAD_ac"] = np.abs(used_DF["CorrectedPAD"])

    DF_long = (used_DF
        .loc[:, ["SID", "Age"] + config.label_cols + ["Type", "PAD", "PAD_ac"]]
        .melt(
            id_vars = ["SID", "Age"] + config.label_cols + ["Type"], 
            value_vars = ["PAD", "PAD_ac"], 
            var_name = "PAD_type", 
            value_name = "PAD_value"
        )
        .replace({
            "STRUCTURE": "STR", 
            "FUNCTIONAL": "FUN"
        })
    )
    DF_long = DF_long.sort_values(by=["Sex", "AgeGroup"], ascending=False)
    if config.sep_sex:
        DF_long["AgeSex"] = DF_long["AgeGroup"] + "_" + DF_long["Sex"]

    return DF_long

## Function: --------------------------------------------------------------------------

def plot_PAD_bars(DF_long, x_lab, output_path, overwrite=False):
    '''
    Plot the PAD values.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style="whitegrid")
        sns.set_context("talk", font_scale=1.2)
        g = sns.catplot(
            data=DF_long, kind="bar",
            x=x_lab, y="PAD_value", hue="PAD_type", col="Type", dodge=True, 
            errorbar="se", palette="dark", alpha=.6, height=6
        )
        g.set_axis_labels("", "PAD Value")
        plt.savefig(output_path)
        print(f"\nBar plot of the PAD values is saved to\n{output_path}")
        plt.close()

## Function set: ----------------------------------------------------------------------

def formated_corr(x, y):
    r, p = pearsonr(x, y, alternative='two-sided')
    if p < .001:
        return f"{r:.2f}***"
    elif p < .01:
        return f"{r:.2f}**"
    elif p < .05:
        return f"{r:.2f}*"
    else:
        return f"{r:.2f}"
    
def plot_cormat(sub_df_wide, ori_types, output_path, overwrite=False, 
                font_scale=1.1, figsize=(3, 3), dpi=200):
    '''
    Plot the correlation matrix for sub-dataframes.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style='white', font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) 
        cormat = sub_df_wide.corr()
        mask = np.zeros_like(cormat)
        mask[np.triu_indices_from(mask)] = True
        annot_mat = cormat.copy(deep=True)
        annot_mat = annot_mat.astype(object)
        annot_mat.iloc[:, :] = ""
        for t1, t2 in itertools.combinations(ori_types, 2):
            annot_mat.loc[t1, t2] = formated_corr(sub_df_wide[t1], sub_df_wide[t2])
            annot_mat.loc[t2, t1] = annot_mat.loc[t1, t2] 
        sns.heatmap(
            cormat, mask=mask, # square=True, 
            vmin=-1, vmax=1, cmap="RdBu_r", 
            cbar=False, # cbar_kws={"shrink": 0.5, "label": "$r$"}, 
            # annot=True, fmt=".2f", # annot_kws={"size": 16}, 
            annot=pd.DataFrame(annot_mat), fmt = "", 
            linewidth=.5
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nCorrelation matrix is saved to\n{output_path}")
        plt.close()

## Function: --------------------------------------------------------------------------

def pairwise_corr(sub_df_dict, excel_file, overwrite=False):
    '''
    Calculate pairwise correlations and save to an .xlsx file.
    <returns>:
    - corr_DF: DataFrame with pairwise correlations.
    '''
    pw_corr_list = []

    for group_name, sub_df in sub_df_dict.items():
        pw_corr = (
            pg.pairwise_corr(sub_df, padjust='bonf')
            .sort_values(by=['p-unc'])[['X', 'Y', 'n', 'r', 'CI95%', 'p-unc', 'p-corr']]
        )
        if (not os.path.exists(excel_file)) or overwrite:
            if not os.path.exists(excel_file):
                pw_corr.to_excel(excel_file, sheet_name=group_name, index=False)
            else:
                with pd.ExcelWriter(excel_file, mode='a') as writer: 
                    pw_corr.to_excel(writer, sheet_name=group_name, index=False)
            print(f"\nPairwise comparisons is saved to\n{excel_file}")

        pw_corr.insert(0, "AgeSex", group_name)
        pw_corr_list.append(pw_corr)
    
    corr_DF = pd.concat(pw_corr_list, ignore_index=True)

    return corr_DF

## Function set: ----------------------------------------------------------------------

def print_p(p):
    if p < .001:
        return "p < .001***"
    elif p < .01:
        return f"p = {p:.3f}**".lstrip('0')
    elif p < .05:
        return f"p = {p:.3f}*".lstrip('0')
    elif p == 1:
        return "p = 1"
    else:
        return f"p = {p:.3f}".lstrip('0')
    
def plot_corr_scatter(corr_DF, sub_df, group_name, t1, t2, p_apply, output_path, overwrite=False, 
                      font_scale=1.2, figsize=(5, 5), dpi=500):
    '''
    Plot the correlation scatter plot for sub-dataframes.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style='whitegrid', font_scale=font_scale)
        fig = plt.figure(num=None, figsize=figsize, dpi=dpi)
        g = sns.JointGrid(
            data=sub_df, x=t1, y=t2, height=5, ratio=3
        )
        g = g.plot_joint(
            sns.regplot, x_jitter=False, y_jitter=False, 
            scatter_kws={'alpha': 0.5, 'edgecolor': 'white'}, line_kws={'linewidth': 1}
        )
        g = g.plot_marginals(
            sns.histplot, kde=True, linewidth=1, bins = 15 # binwidth=2
        )
        g.refline(x=0, color='g')
        g.refline(y=0, color='g')

        sub_corr_df = corr_DF.query("AgeSex == @group_name & X == @t1 & Y == @t2")
        if sub_corr_df.empty:
            sub_corr_df = corr_DF.query("AgeSex == @group_name & X == @t2 & Y == @t1")
            
        N = sub_corr_df["n"].iloc[0]
        r = sub_corr_df["r"].iloc[0]
        p = sub_corr_df[p_apply].iloc[0]
        p_print = print_p(p).replace("p", "p-unc") if p_apply == "p-unc" else print_p(p)

        plt.suptitle(f"r = {r:.2f}, {p_print}, N = {N:.0f}")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nCorrelation plot is saved to\n{output_path}")
        plt.close()

## Main function: ---------------------------------------------------------------------

def main():
    config = Config()
    config = load_parameters(config)
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    ## Load data:
    selected_cols = [config.sid_name, "Age", "PredictedAgeDifference", "CorrectedPAD"]
    DF = load_data(config, selected_cols)

    ## Aggregate model types and feature numbers (save to a table):
    save_model_info(
        DF, config.label_cols,
        os.path.join(config.output_folder, config.model_info_filename), 
        overwrite=args.overwrite
    )

    ## Aggregate median and STD of the data (save to a table):
    part_DF, stats_table = save_discriptive_table(
        DF, os.path.join(config.output_folder, config.disc_table_filename)
    )

    ## Down-sample or bootstrap the data if specified:
    if args.downsample or args.bootstrap:
        sampling_method = "down-sampling" if args.downsample else "bootstrapping"
        if args.seed is not None:
            seed = args.seed
        else:
            seed = np.random.randint(0, 10000)
        sub_folder = os.path.join(
            config.output_folder, f"{sampling_method} (N={args.sample_size}; seed={seed})")
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        balanced_DF = make_balanced_dataframe(
            part_DF, stats_table, seed, 
            os.path.join(sub_folder, config.balanced_disc_table_fn)
        )
        if config.sep_sex:
            merge_on = [config.sid_name, "Age", "Sex", "AgeGroup"]
        else:
            merge_on = [config.sid_name, "Age", "AgeGroup"]
        used_DF = balanced_DF.merge(DF, on=merge_on, how="left")
    else:
        used_DF = DF
        sub_folder = config.output_folder

    ## Modify DataFrame and transform to long format:
    DF_long = modify_DF(used_DF, config)
    
    ## Plot of PAD bars: 
    x_lab = "AgeSex" if config.sep_sex else "AgeGroup"
    plot_PAD_bars(
        DF_long, x_lab, os.path.join(sub_folder, config.barplot_filename), 
        overwrite=args.overwrite
    )
    
    ## Specify PAD type:
    pad_type = "PAD" if args.use_pad else "PAD_ac"

    ## Create an index column for pivoting:
    if args.bootstrap or ("SID" not in DF_long.columns) or ("Age" not in DF_long.columns):
        DF_long["idx"] = (
            DF_long
            .query("PAD_type == @pad_type")
            .groupby(["Type", "AgeSex"])
            .cumcount()
        )
        idx_cols = ["idx"]
    else:
        idx_cols = ["SID", "Age"]

    ## Pivot data for calculating correlations:
    sub_df_dict = {}
    for (age_group, sex) in config.label_list:
        group_name = f"{age_group}_{sex}"
        sub_df = DF_long.query("AgeSex == @group_name & PAD_type == @pad_type")        
        sub_df_wide = (sub_df
            .loc[:, idx_cols + ["Type", "PAD_value"]]
            .pivot(index = idx_cols, columns = "Type", values = "PAD_value")
            .reset_index(drop=True)
        )
        sub_df_dict[group_name] = sub_df_wide

        ## Plot correlation matrices:
        plot_cormat(
            sub_df_wide, set(DF_long["Type"]), os.path.join(
                sub_folder, config.cormat_fn_template.replace("GroupName", group_name)),
            figsize=(3, 3) if len(config.feature_orientations) <= 3 else (4, 4),
            overwrite=args.overwrite
        )
        
    ## Calculate pairwise correlations (save to different sheets in an .xlsx file):
    corr_DF = pairwise_corr(
        sub_df_dict, os.path.join(sub_folder, config.corr_table_filename), 
        overwrite=args.overwrite
    )

    ## Plot correlations (scatter plots):
    p_apply = "p-corr" if args.p_adjust else "p-unc"

    for group_name, sub_df in sub_df_dict.items():
        for t1, t2 in [("BEH", "FUN"), ("BEH", "STR"), ("FUN", "STR")]:
            plot_corr_scatter(
                corr_DF, sub_df, group_name, t1, t2, p_apply, 
                os.path.join(sub_folder, config.scatter_fn_template.replace("GroupName", group_name).replace("Type1", t1).replace("Type2", t2)), 
                overwrite=args.overwrite
            )

## Finally: ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
    print("\nDone!\n")