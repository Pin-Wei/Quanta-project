#!/usr/bin/python

# python gen_derivatives.py -f FOLDER_NAME 

import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

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
    7) Sunburst plots visualizing the proportion of features domain and approach used
      in each model: 
        '[pie] {ori_type}.png'.

    The files will be saved in a folder with the same name as the '-f' argument
    under the 'derivatives' folder.
    """
)
parser.add_argument("-f", "--folder", type=str, required=True, 
                    help="The folder name of the model outputs.")
parser.add_argument("-pad", "--use_pad", action="store_true", default=False,
                    help="Use PAD values that have not been corrected for age.")
# parser.add_argument("-b", "--bootstrap", action="store_true", default=False, 
#                     help="Model was trained on bootstrapped data.")
parser.add_argument("-ia", "--ignore_all", action="store_true", default=False,
                    help="Ignore 'All' feature orientations.")
parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
                    help="Overwrite if the output files already exist.")
parser.add_argument("-pa", "--p_adjust", action="store_true", default=False,
                    help="Use adjusted p-values for pairwise correlations.")
args = parser.parse_args()

## Classes: ---------------------------------------------------------------------------

class Config:
    def __init__(self):
        self.folder = args.folder
        self.input_folder = os.path.join("outputs", self.folder)
        self.output_folder = os.path.join("derivatives", self.folder)
        ## Tables: 
        self.model_info_filename     = "[table] model types and feature numbers.csv"
        self.disc_table_filename     = "[table] median and std of ages.csv"
        self.balanced_disc_table_fn  = "[table] median and std of ages after balancing.csv"
        self.pad_corr_table_filename = "[table] pairwise correlations between PAD_ac.xlsx"
        ## Figures: 
        self.data_hist_fn_template   = "[hist] distributions of real and synthetic data in <GroupName>.png"
        self.data_cormat_fn_template = "[cormat] between features in <GroupName>'s <S_or_R> data.png"
        self.pad_barplot_filename    = "[bar] PAD values.png"
        self.pad_cormat_fn_template  = "[cormat] between PAD_ac in <GroupName>.png"
        self.pad_scatter_fn_template = "[scatter] between PAD_ac in <GroupName> (<Type1> × <Type2>).png"
        self.sunburst_fn_template    = "[pie] <FeatureType>.png"

class Description:
    def __init__(self):
        self.domain_approach_mapping = {
            "STRUCTURE": {
                "domains": ["STRUCTURE"],
                "approaches": ["MRI"]
            }, 
            "BEH": {
                "domains": ["MOTOR", "MEMORY", "LANGUAGE"],
                "approaches": ["BEH"]
            },
            "FUNCTIONAL": {
                "domains": ["MOTOR", "MEMORY", "LANGUAGE"],
                "approaches": ["EEG", "MRI"]
            }, 
            "ALL": {
                "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
                "approaches": ["MRI", "BEH", "EEG"]
            }
        }   

## Functions: -------------------------------------------------------------------------

def load_description(config, desc):
    '''
    Load the parameters from the description file and update the description object.
    <returns>:
    - desc: Updated description object.
    '''
    ## Load description file: 
    desc_path = os.path.join(config.input_folder, "description.json")
    with open(desc_path, 'r', errors='ignore') as f:
        desc_json = json.load(f)

    ## Data groups:
    desc.sep_sex = bool(desc_json["SexSeparated"]) 
    desc.age_group_labels = desc_json["AgeGroups"]
    desc.age_breaks = (
        [ 0 ] + 
        [ int(x.split("-")[0]) for x in desc.age_group_labels[1:-1] ] + 
        [ int(desc.age_group_labels[-1].split("-")[-1]) ] + 
        [ np.inf ]
    )
    desc.label_cols = ["AgeGroup", "Sex"] if desc.sep_sex else ["AgeGroup"]
    if desc.sep_sex:
        desc.label_list = list(itertools.product(desc.age_group_labels, ["M", "F"]))
    else:
        desc.label_list = desc.age_group_labels   

    ## Used feature orientations (Options: ["STRUCTURE", "BEH", "FUNCTIONAL", "ALL"]):
    desc.feature_orientations = desc_json["FeatureOrientations"] 
    if args.ignore_all:
        desc.feature_orientations = ["STRUCTURE", "BEH", "FUNCTIONAL"]

    ## If testset ratio is 0, then the data was not split into training and testing sets:
    desc.traintest = True if desc_json["TestsetRatio"] != 0 else False
    desc.sid_name = "TestingSubjID" if desc.traintest else "SubjID"

    ## Whether the data was synthetized:
    desc.data_synthetized = True if desc_json["DataBalancingMethod"] == "SMOTENC" else False

    return desc


def load_data(config, desc, selected_cols):
    '''
    Load the model results from the input folder.
    <returns>:
    - DF: DataFrame with selected columns.
    - selected_features: Dictionary with selected features for each feature orientation.
    '''
    print("\nLoading data...")

    selected_result_list = []
    selected_features = { o: {} for o in desc.feature_orientations }
    training_subj_infos = []

    for label in desc.label_list:
        if desc.sep_sex:
            age_group, sex = label
            group_name = f"{age_group}_{sex}"
        else:
            age_group = label
            group_name = label

        for ori_name in desc.feature_orientations:
            data_path = os.path.join(
                config.input_folder, f"results_{group_name}_{ori_name}.json")
            
            if os.path.exists(data_path):
                print(os.path.basename(data_path))

                with open(data_path, 'r', errors='ignore') as f:
                    results = json.load(f)
                    selected_results = pd.DataFrame({ 
                        k: v for k, v in results.items() if k in selected_cols
                    })
                    selected_results["AgeGroup"] = age_group
                    if desc.sep_sex:
                        selected_results["Sex"] = sex
                    selected_results["Type"] = ori_name
                    selected_result_list.append(selected_results)

                    selected_features[ori_name][group_name] = results["FeatureNames"]

        if desc.traintest:            
            temp_DF = pd.DataFrame({
                "SID": [ x.replace("sub-0", "") for x in results["TrainingSubjID"] ]
            })
            temp_DF["AgeGroup"] = age_group
            training_subj_infos.append(temp_DF)

    DF = pd.concat(selected_result_list, ignore_index=True)
    DF["SID"] = DF[desc.sid_name].map(lambda x: x.replace("sub-0", ""))
    DF.drop(columns=[desc.sid_name], inplace=True)

    if desc.traintest: 
        training_subj_infos = pd.concat(training_subj_infos, ignore_index=True)
        preprocessed_data = pd.read_csv(os.path.join(config.input_folder, "preprocessed_data.csv"))
        all_subj_infos = (
            preprocessed_data
            .loc[:, ["ID", "BASIC_INFO_AGE", "BASIC_INFO_SEX"]]
            .rename(columns={
                "BASIC_INFO_AGE": "Age", 
                "BASIC_INFO_SEX": "Sex"
            })
        )
        all_subj_infos["Sex"] = all_subj_infos["Sex"].map({1: "M", 2: "F"})
        all_subj_infos["SID"] = all_subj_infos["ID"].map(lambda x: x.replace("sub-0", ""))
        all_subj_infos.drop(columns=["ID"], inplace=True)
        DF_train_info = pd.merge(training_subj_infos, all_subj_infos, on="SID", how='inner')
    else:
        DF_train_info = None

    return DF, selected_features, DF_train_info


def plot_data_dist(DF_marked, selected_features, age_group, sex, output_path, 
                   orientations=["STRUCTURE", "BEH", "FUNCTIONAL"],
                   bins=30, density=False, alpha=0.8, overwrite=False):
    '''
    Plot data distribution for selected features to compare real and synthetic data.
    <returns>:
    - targ_col_list: List of target columns used for plotting.
    '''
    group_name = f"{age_group}_{sex}"
    targ_col_list = []
    DRAW_FIG = True if (not os.path.exists(output_path)) or overwrite else False

    if DRAW_FIG: 
        plt.style.use('seaborn-v0_8-white')
        fig = plt.figure(figsize=(12, 10))
        handles, labels = [], []
    
    for ori_idx, ori_name in enumerate(orientations):
        for targ_idx in range(2):
            targ_col = selected_features[ori_name][group_name][targ_idx]
            targ_col_list.append(targ_col)

            if DRAW_FIG: 
                targ_vals_S = DF_marked.query(
                    "R_S == 'Synthetic' & AGE_GROUP == @age_group & SEX == @sex"
                )[targ_col]
                targ_vals_R = DF_marked.query(
                    "R_S == 'Real' & AGE_GROUP == @age_group & SEX == @sex"
                )[targ_col]
                
                fig_idx = ori_idx * 2 + targ_idx + 1
                ax = plt.subplot(len(orientations), 2, fig_idx)
                ax.hist(
                    [targ_vals_S, targ_vals_R], 
                    bins=bins, density=density, 
                    color=['#3399FF', '#FF9933'], alpha=alpha, 
                    label=[f"S ({len(targ_vals_S)})", f"R ({len(targ_vals_R)})"]
                )
                ax.set_xlabel(targ_col, fontsize=12)
                ax.set_ylabel("")
                handles.append(ax.get_legend_handles_labels()[0])
                labels = ax.get_legend_handles_labels()[1]
    if DRAW_FIG: 
        fig.suptitle(f"{age_group}, {sex}", fontsize=18)
        plt.tight_layout()     
        plt.figlegend(sum(handles, []), labels, bbox_to_anchor=(1.1, 0.55), fontsize=12)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\nDistribution plots is saved to:\n{output_path}")

    return targ_col_list


def save_model_info(DF, label_cols, output_path, overwrite=False):
    '''
    Save the model types and feature numbers to a .csv table.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        DF["Info"] = DF.apply(
            lambda x: f"{x['Model']}_{x['NumberOfFeatures']}", axis=1
        )
        model_info = (DF
            .loc[:, label_cols + ["Type", "Info"]]
            .drop_duplicates()
            .pivot(index = label_cols, 
                   columns = "Type", 
                   values = "Info")
        )
        model_info.to_csv(output_path)
        print(f"\nModel types and feature numbers are saved to:\n{output_path}")


def save_discriptive_table(DF, output_path, overwrite=False):
    '''
    Save the median and standard deviation of the data to a .csv table.
    # <previous returns>:
    # - DF_info: DataFrame with unique subjects.
    # - stats_table: Table with median and standard deviation of the data.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        DF_info = (DF
            .loc[:, ["SID", "Age", "Sex", "AgeGroup"]]
            .drop_duplicates("SID")
        )
        stats_table = (DF_info
            .groupby(["Sex", "AgeGroup"])["Age"]
            .agg(["count", "median", "std"])
            .rename(columns = {"count": "N", "median": "Median", "std": "STD"})
            .reset_index()
        )
        stats_table.to_csv(output_path, index=False)
        print(f"\nData median and std are saved to:\n{output_path}")


def modify_DF(DF, desc):
    '''
    Modify the DataFrame and transform it to long format.
    <returns>:
    - DF_long: DataFrame in long format.
    '''
    DF.rename(columns={
        "PredictedAgeDifference": "PAD", 
        "CorrectedPAD": "PAD_ac"
    }, inplace=True)

    DF_long = (DF
        .loc[:, ["SID", "Age"] + desc.label_cols + ["Type", "PAD", "PAD_ac"]]
        .melt(
            id_vars = ["SID", "Age"] + desc.label_cols + ["Type"], 
            value_vars = ["PAD", "PAD_ac"], 
            var_name = "PAD_type", 
            value_name = "PAD_value"
        )
        .replace({
            "STRUCTURE": "STR", 
            "FUNCTIONAL": "FUN"
        })
    )
    DF_long["PAD_abs_value"] = DF_long["PAD_value"].abs()

    DF_long = DF_long.sort_values(by=["Sex", "AgeGroup"], ascending=False)
    if desc.sep_sex:
        DF_long["AgeSex"] = DF_long["AgeGroup"] + "_" + DF_long["Sex"]

    return DF_long


def plot_pad_bars(DF_long, x_lab, output_path, overwrite=False):
    '''
    Plot the PAD values.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style="whitegrid")
        sns.set_context("talk", font_scale=1.2)
        g = sns.catplot(
            data=DF_long, kind="bar",
            x=x_lab, y="PAD_abs_value", hue="PAD_type", col="Type", dodge=True, 
            errorbar="se", palette="dark", alpha=.6, height=6
        )
        g.set_axis_labels("", "PAD Value")
        plt.savefig(output_path)
        print(f"\nBar plot of the PAD values is saved to:\n{output_path}")
        plt.close()


# def rename_cols(col_list):
#     renamed_col_list = []

#     for col in col_list:
#         if "STRUCTURE" in col:
#             col_name = "STR"
#         elif "BEH" in col:
#             col_name = "BEH"
#         else:
#             if "EEG" in col:
#                 col_name = "EEG"
#             elif "MEG" in col:
#                 col_name = "MEG"
#             elif "MRI" in col:
#                 col_name = "fMRI"
          
#         if "WM" in col:
#             col_name += " (WM)"
#         elif "GM" in col:
#             col_name += " (GM)"
#         elif "MEMORY" in col:
#             col_name += " (memory)"
#         elif "MOTOR" in col:
#             col_name += " (motor)"
#         elif "LANGUAGE" in col:
#             col_name += " (language)"

#         renamed_col_list.append(col_name)

#     return renamed_col_list


def rename_cols(col_list):
    renamed_col_list = []
    for x, col in enumerate(col_list):
        renamed_col_list.append(f"({col}) #{x+1}")
    return renamed_col_list


def format_r(x, y):
    r, p = pearsonr(x, y, alternative='two-sided')
    if p < .001:
        return f"{r:.2f}***"
    elif p < .01:
        return f"{r:.2f}**"
    elif p < .05:
        return f"{r:.2f}*"
    else:
        return f"{r:.2f}"
    

def plot_cormat(wide_DF, targ_cols, output_path, 
                targ_col_names=None, xr=0, yr=0, 
                overwrite=False, font_scale=1.1, figsize=(3, 3), dpi=200):
    '''
    Plot the correlation matrix for sub-dataframes.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style='white', font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) 
        cormat = wide_DF.corr()
        mask = np.zeros_like(cormat)
        mask[np.triu_indices_from(mask)] = True
        annot_mat = cormat.copy(deep=True) # ensure that the columns are the same
        annot_mat = annot_mat.astype(object)
        annot_mat.iloc[:, :] = ""
        for t1, t2 in itertools.combinations(targ_cols, 2):
            annot_mat.loc[t1, t2] = format_r(wide_DF[t1], wide_DF[t2])
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
        if targ_col_names is not None:
            ax.set_yticklabels(targ_col_names, rotation=yr)
            # ax.set_xticklabels(targ_col_names, rotation=xr)
            ax.set_xticklabels([ f"#{x+1}" for x in range(len(targ_col_names)) ], rotation=xr)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nCorrelation matrix is saved to:\n{output_path}")
        plt.close()


def calc_pad_corr_table(sub_df_dict, excel_file, overwrite=False):
    '''
    Calculate pairwise correlations and save to an .xlsx file.
    <returns>:
    - corr_DF: DataFrame with pairwise correlations.
    '''
    if overwrite and os.path.exists(excel_file):
        os.remove(excel_file)
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
            print(f"\nPairwise comparisons is saved to:\n{excel_file}")

        pw_corr.insert(0, "AgeSex", group_name)
        pw_corr_list.append(pw_corr)
    
    corr_DF = pd.concat(pw_corr_list, ignore_index=True)

    return corr_DF


def format_p(p):
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
    

def plot_pad_corr_scatter(corr_DF, sub_df, group_name, t1, t2, p_apply, output_path, 
                          overwrite=False, font_scale=1.2, figsize=(5, 5), dpi=500):
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
        p_print = format_p(p).replace("p", "p-unc") if p_apply == "p-unc" else format_p(p)

        plt.suptitle(f"r = {r:.2f}, {p_print}, N = {N:.0f}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"\nCorrelation plot is saved to:\n{output_path}")


def make_feature_DF(ori_name, feature_list, domain_approach_mapping):
    '''
    Make a DataFrame containing the domain and approach labels for each feature, 
    as well as the proportion information of each domain and approach.
    '''
    ## Add domain and approach information:
    if ori_name == "STRUCTURE":
        domains = ["GM", "WM"]
    else:
        domains = domain_approach_mapping[ori_name]["domains"]
    approaches = domain_approach_mapping[ori_name]["approaches"]
    
    dict_list = []    
    for feature in feature_list:
        for domain in domains:
            if domain in feature:
                for approach in approaches: 
                    if approach in feature:
                        dict_list.append({
                            "domain": domain, "approach": approach, "feature": feature
                        })
    feature_DF = pd.DataFrame(dict_list)

    ## Add percentage information:
    main_pr = feature_DF["approach"].value_counts(normalize=True) * 100
    feature_DF["approach_pr"] = feature_DF["approach"].map(main_pr)
    feature_DF["approach_and_pr"] = feature_DF.apply(
        lambda row: f"{row['approach']}<br>({row['approach_pr']:.1f}%)"
        if row['approach_pr'] > 7 else f"{row['approach']} ({row['approach_pr']:.1f}%)", 
        axis=1
    )
    sub_pr = feature_DF.groupby("approach")["domain"].value_counts(normalize=True) * 100
    feature_DF["domain_pr"] = feature_DF.apply(
        lambda row: sub_pr[row["approach"]][row["domain"]], axis=1
    )
    feature_DF["overall_pr"] = feature_DF.apply(
        lambda row: sub_pr[row["approach"]][row["domain"]] * row["approach_pr"], axis=1
    )
    feature_DF["domain_and_pr"] = feature_DF.apply(
        lambda row: f"{row['domain']}<br>({row['domain_pr']:.1f}%)" 
        if row['overall_pr'] > 300 else f"{row['domain']} ({row['domain_pr']:.1f}%)", 
        axis=1
    )
    
    return feature_DF


def plot_feature_sunbursts(feature_DF_dict, fig_title, subplot_annots, output_path, 
                           overwrite=False, ncol=None, nrow=None):
    '''
    Plot the sunburst plots for each dataframe in the dictionary, 
    whose keys are the group names.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        if ncol is None:
            ncol = 2
        if nrow is None:
            nfig = len(feature_DF_dict.keys())
            nrow = (nfig + ncol - 1) // ncol
        fig = make_subplots(
            rows=nrow, cols=ncol, specs=[[{'type': 'domain'}] * ncol] * nrow
        )
        annotations = []
        for x, (group_name, feature_DF) in enumerate(feature_DF_dict.items()):
            c = (x % 2) + 1
            r = (x // 2) + 1
            sunburst_fig = px.sunburst(
                feature_DF, path=["approach_and_pr", "domain_and_pr"], 
                maxdepth=2, width=500, height=500
            )
            fig.add_trace(
                sunburst_fig.data[0], row=r, col=c
            )
            annotations.append(dict(
                x=[0.05, 0.95][c-1], y=[0.5, -0.1][r-1], xref="paper", yref="paper", 
                showarrow=False, font=dict(size=32), 
                text=subplot_annots[group_name]
            ))
        fig.update_layout(
            grid=dict(columns=ncol, rows=nrow), 
            margin=dict(l=0, r=0, t=10, b=100), 
            title_text=fig_title, 
            title_x=0.5, title_y=0.5, font=dict(size=40), 
            template="plotly_white", 
            width=500*ncol, height=400*nrow, 
            annotations=annotations
        )
        plt.tight_layout()
        pio.write_image(fig, output_path, format="png", scale=2)
        plt.close()
        print(f"\nSunburst plot is saved to:\n{output_path}")
        
## Main: ------------------------------------------------------------------------------

def main():
    config = Config()
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    desc = Description()
    desc = load_description(config, desc)

    ## Load data:
    selected_cols = [
        desc.sid_name, "Age", "PredictedAgeDifference", "CorrectedPAD", 
        "Model", "NumberOfFeatures"
    ]
    DF, selected_features, DF_train_info = load_data(config, desc, selected_cols)

    ## If the data was synthetized, load the marked data:
    if desc.data_synthetized:
        DF_marked = pd.read_csv(
            os.path.join(config.input_folder, "preprocessed_data (marked).csv")
        )
        DF_marked["SEX"] = DF_marked["BASIC_INFO_SEX"].replace({1: "M", 2: "F"})
        DF_marked["AGE_GROUP"] = pd.cut(DF_marked["BASIC_INFO_AGE"], bins=desc.age_breaks, labels=desc.age_group_labels)
        
        for label in desc.label_list:
            age_group, sex = label

            ## Compare the distribution between real and synthetic data for selected features:
            fn = config.data_hist_fn_template.replace("<GroupName>", f"{age_group}_{sex}")
            targ_col_list = plot_data_dist(
                DF_marked, selected_features, age_group, sex, 
                os.path.join(config.output_folder, fn)
            )

            ## Plot correlation matrices for selected features:
            for S_or_R in ["Synthetic", "Real"]:
                fn = config.data_cormat_fn_template.replace("<GroupName>", f"{age_group}_{sex}").replace("<S_or_R>", S_or_R)
                plot_cormat(
                    wide_DF=DF_marked.query("R_S == @S_or_R & AGE_GROUP == @age_group & SEX == @sex").loc[:, targ_col_list], 
                    targ_cols=targ_col_list, 
                    output_path=os.path.join(config.output_folder, fn), 
                    targ_col_names=rename_cols(targ_col_list), 
                    # targ_col_names=targ_col_list, 
                    figsize=(12, 4), 
                    overwrite=args.overwrite
                )

    ## Aggregate model types and feature numbers (save to a table):
    save_model_info(
        DF, desc.label_cols,
        os.path.join(config.output_folder, config.model_info_filename), 
        overwrite=args.overwrite
    )

    ## Aggregate medians and STDs of ages for each participant group (save to a table):
    save_discriptive_table(
        DF, os.path.join(config.output_folder, config.disc_table_filename), 
        overwrite=args.overwrite
    )
    
    if desc.traintest:
        fn = config.disc_table_filename.replace(".csv", " (training set).csv")
        save_discriptive_table(
            DF_train_info, os.path.join(config.output_folder, fn)
        )

    ## Modify DataFrame and transform it to long format:
    DF_long = modify_DF(DF, desc)
    
    ## Plot PAD bars: 
    x_lab = "AgeSex" if desc.sep_sex else "AgeGroup"
    plot_pad_bars(
        DF_long, x_lab, os.path.join(config.output_folder, config.pad_barplot_filename), 
        overwrite=args.overwrite
    )
    if args.ignore_all:
        fn = config.pad_barplot_filename.replace(".png", " (ignore 'All').png")
        plot_pad_bars(
            DF_long, x_lab, os.path.join(config.output_folder, fn), 
            overwrite=True
        )
    
    ## Specify PAD type:
    pad_type = "PAD" if args.use_pad else "PAD_ac"

    ## Create an index column for pivoting:
    # if args.bootstrap or ("SID" not in DF_long.columns) or ("Age" not in DF_long.columns):
    if ("SID" not in DF_long.columns) or ("Age" not in DF_long.columns):
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
    for (age_group, sex) in desc.label_list:
        group_name = f"{age_group}_{sex}"
        sub_df = DF_long.query("AgeSex == @group_name & PAD_type == @pad_type")        
        sub_df_wide = (sub_df
            .loc[:, idx_cols + ["Type", "PAD_value"]]
            .pivot(index = idx_cols, columns = "Type", values = "PAD_value")
            .reset_index(drop=True)
        )
        sub_df_dict[group_name] = sub_df_wide

        ## Plot correlation matrices (without 'All' feature orientation):
        fn = config.pad_cormat_fn_template.replace("<GroupName>", group_name)
        plot_cormat(
            wide_DF=sub_df_wide, 
            targ_cols=set(DF_long["Type"]), 
            output_path=os.path.join(config.output_folder, fn), 
            figsize=(3, 3) if len(desc.feature_orientations) <= 3 else (4, 4),
            overwrite=args.overwrite
        )

        ## Plot correlation matrices (with 'All' feature orientation):
        if args.ignore_all:
            fn2 = fn.replace(".png", " (ignore 'All').png")
            plot_cormat(
                wide_DF=sub_df_wide, 
                targ_cols=set(DF_long["Type"]), 
                output_path=os.path.join(config.output_folder, fn2),
                figsize=(3, 3), overwrite=True
            )
        
    ## Calculate pairwise correlations (save to different sheets in an .xlsx file):
    corr_DF = calc_pad_corr_table(
        sub_df_dict, os.path.join(config.output_folder, config.pad_corr_table_filename), 
        overwrite=args.overwrite
    )

    ## Plot correlations (scatter plots):
    p_apply = "p-corr" if args.p_adjust else "p-unc"

    for group_name, sub_df in sub_df_dict.items():
        for t1, t2 in [("BEH", "FUN"), ("BEH", "STR"), ("FUN", "STR")]:
            plot_pad_corr_scatter(
                corr_DF, sub_df, group_name, t1, t2, p_apply, 
                os.path.join(config.output_folder, config.pad_scatter_fn_template.replace("<GroupName>", group_name).replace("<Type1>", t1).replace("<Type2>", t2)), 
                overwrite=args.overwrite
            )
    
    ## Prepare a smaller, non-duplicatie dataframe:
    model_info_DF = (
        DF.loc[:, desc.label_cols + ["Type", "Model", "NumberOfFeatures"]]
        .drop_duplicates()
    )

    ## Make a dataframe with hierachical labels for selected features:
    for ori_name in desc.feature_orientations:                

        ## One dataframe per group:
        feature_DF_dict = {
            group_name: make_feature_DF(
                ori_name, feature_list, desc.domain_approach_mapping
            ) 
            for group_name, feature_list in selected_features[ori_name].items()
        }

        ## Prepare annotation for each subplot:
        subplot_annots = {}
        for (age_group, sex) in desc.label_list: 
            group_name = f"{age_group}_{sex}"
            model_info = model_info_DF.query("Type == @ori_name & AgeGroup == @age_group & Sex == @sex")
            model_type = model_info['Model'].iloc[0]
            n_features = model_info['NumberOfFeatures'].iloc[0]
            subplot_annots[group_name] = f"{group_name} ({model_type} - {n_features})" 
                
        ## One set of sunburst charts per feature type:
        plot_feature_sunbursts(
            feature_DF_dict, f"{ori_name[:3]}", subplot_annots, os.path.join(
                config.output_folder, config.sunburst_fn_template.replace("<FeatureType>", ori_name[:3])),
            overwrite=args.overwrite
        )

## Finally: ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
    print("\nDone!\n")