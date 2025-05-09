#!/usr/bin/python

# python gen_derivatives.py -f FOLDER_NAME 

import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr, entropy 

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from utils import standardized_feature_list, domain_approach_mapping_dict

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
parser.add_argument("-ia", "--ignore_all", action="store_true", default=False,
                    help="Ignore 'All' feature orientations.")
parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
                    help="Overwrite if the output files already exist.")
parser.add_argument("-pa", "--p_adjust", action="store_true", default=False,
                    help="Use adjusted p-values for pairwise correlations.")
args = parser.parse_args()

## Classes and Functions: -------------------------------------------------------------

class Config:
    def __init__(self):
        self.folder = args.folder
        self.input_folder = os.path.join("outputs", self.folder)
        self.output_folder = os.path.join("derivatives", self.folder)
        self.pad_type = "PAD" if args.use_pad else "PAD_ac"
        ## Tables: 
        self.model_info_filename     = "[table] model types and feature numbers.csv"
        self.disc_table_filename     = "[table] median and std of ages.csv"
        self.balanced_disc_table_fn  = "[table] median and std of ages after balancing.csv"
        self.pad_corr_table_filename = f"[table] pairwise correlations between {self.pad_type}.xlsx"
        ## Figures: 
        self.data_hist_fn_template   = "[hist] distributions of real and synthetic data in <GroupName>.png"
        self.data_cormat_fn_template = "[cormat] between features in <GroupName>'s <S_or_R> data.png"
        self.pad_barplot_filename    = "[bar] PAD values.png"
        self.pad_cormat_fn_template  = f"[cormat] between {self.pad_type} in <GroupName>.png"
        self.pad_scatter_fn_template = f"[scatter] between {self.pad_type} in <GroupName> (<Type1> × <Type2>).png"
        self.sunburst_fn_template    = "[pie] <FeatureType>.png"

def load_description(config):
    '''
    Load the parameters from the description file and update the description object.
    <returns>:
    - desc: Updated description object.
    '''
    ## Load description file: 
    desc_path = os.path.join(config.input_folder, "description.json")
    with open(desc_path, 'r', errors='ignore') as f:
        desc_json = json.load(f)

    class Description:
        def __init__(self):
            ## Data groups:
            self.sep_sex = bool(desc_json["SexSeparated"]) 
            self.age_group_labels = desc_json["AgeGroups"]
            self.age_breaks = (
                [ 0 ] + 
                [ int(x.split("-")[0]) for x in self.age_group_labels[1:-1] ] + 
                [ int(self.age_group_labels[-1].split("-")[-1]) ] + 
                [ np.inf ]
            )
            self.label_cols = ["AgeGroup", "Sex"] if self.sep_sex else ["AgeGroup"]
            if self.sep_sex:
                self.label_list = list(itertools.product(self.age_group_labels, ["M", "F"]))
            else:
                self.label_list = self.age_group_labels   

            ## Used feature orientations (options: ["STRUCTURE", "BEH", "FUNCTIONAL", "ALL"]):
            self.feature_orientations = desc_json["FeatureOrientations"] 
            if args.ignore_all:
                self.feature_orientations = ["STRUCTURE", "BEH", "FUNCTIONAL"]

            ## If testset ratio is 0, then the data was not split into training and testing sets:
            self.traintest = True if desc_json["TestsetRatio"] != 0 else False
            self.sid_name = "TestingSubjID" if self.traintest else "SubjID"

            ## Whether the data was synthetized:
            if desc_json["DataBalancingMethod"] in ["CTGAN", "SMOTENC"]:
                self.data_synthetized = True
            else:
                self.data_synthetized = False

    return Description()

def load_data(config, desc, selected_cols):
    '''
    <returns>:
    - data_DF (pd.DataFrame): The dataset used for model training.
    - result_DF (pd.DataFrame): The (JSON) results of model training (only include selected columns).
    - selected_features (dict): The features used by each model.
    - trainset_info_DF (pd.DataFrame): Personal information of participants in the training set .
    '''
    print("\nLoading data...")

    ## The data used for model training:
    if desc.data_synthetized:
        data_DF = pd.read_csv(os.path.join(config.input_folder, "prepared_data (marked).csv"))
    else:
        data_DF = pd.read_csv(os.path.join(config.input_folder, "prepared_data.csv"))
    data_DF["AGE_GROUP"] = pd.cut(data_DF["BASIC_INFO_AGE"], bins=desc.age_breaks, labels=desc.age_group_labels)
    if desc.sep_sex:
        data_DF["SEX"] = data_DF["BASIC_INFO_SEX"].replace({1: "M", 2: "F"})    
    data_DF["SID"] = data_DF["ID"].map(lambda x: x.replace("sub-0", ""))
    data_DF.drop(columns=["ID"], inplace=True)
    
    ## The results of modeling:
    selected_result_list = []
    selected_features = { o: {} for o in desc.feature_orientations }
    trainset_infos = []

    for label in desc.label_list:
        if desc.sep_sex:
            age_group, sex = label
            group_name = f"{age_group}_{sex}"
        else:
            age_group = label
            group_name = label

        ## Load the model results:
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
            trainset_infos.append(temp_DF)

    result_DF = pd.concat(selected_result_list, ignore_index=True)
    result_DF["SID"] = result_DF[desc.sid_name].map(lambda x: x.replace("sub-0", ""))
    result_DF.drop(columns=[desc.sid_name], inplace=True)

    if desc.traintest: 
        trainset_info_DF = pd.concat(trainset_infos, ignore_index=True)
        all_subj_info_DF = (
            data_DF
            .loc[:, ["SID", "BASIC_INFO_AGE", "BASIC_INFO_SEX"]]
            .rename(columns={
                "BASIC_INFO_AGE": "Age", 
                "BASIC_INFO_SEX": "Sex"
            })
        )
        trainset_info_DF = trainset_info_DF.merge(all_subj_info_DF, on="SID", how='inner')
    else:
        trainset_info_DF = None

    return data_DF, result_DF, selected_features, trainset_info_DF

def compute_KL_divergence(targ_vals_R, targ_vals_S, num_bins=30):
    '''
    Compute relative entropy (also known as Kullback-Leibler divergence) between two distributions.
    '''
    hist_bins = np.linspace(
        min(targ_vals_R.min(), targ_vals_S.min()), 
        max(targ_vals_R.max(), targ_vals_S.max()), 
        num_bins + 1
    )
    real_hist, _ = np.histogram(targ_vals_R, bins=hist_bins)
    synth_hist, _ = np.histogram(targ_vals_S, bins=hist_bins)
    D = entropy(
        pk=synth_hist + 1e-10, qk=real_hist + 1e-10
    ) # add a small number to avoid division by zero

    return D

def test_real_vs_synthetic(targ_vals_R, targ_vals_S, observed_D, n_permutations=1000):
    '''
    Use permutation (Monte Carlo) test 
    to statistically assess whether the KL divergence is small enough 
    to conclude that a synthetic dataset matches the original one in distribution. 
    '''
    labels = ['Real'] * len(targ_vals_R) + ['Synthetic'] * len(targ_vals_S)
    combined_data = pd.concat([targ_vals_R, targ_vals_S], ignore_index=True)
    permuted_D_list = []
    for _ in range(n_permutations):
        shuffled_labels = np.random.permutation(labels)
        G1 = combined_data[np.array(shuffled_labels) == 'Real']
        G2 = combined_data[np.array(shuffled_labels) == 'Synthetic']
        D = compute_KL_divergence(G1, G2)
        permuted_D_list.append(D)

    permuted_D_list = np.array(permuted_D_list)
    p_value = (permuted_D_list > observed_D).mean()

    return p_value

def plot_data_dist(data_DF, age_group, sex, selected_features, output_path, 
                   orientations=["STRUCTURE", "BEH", "FUNCTIONAL"],
                   num_bins=30, alpha=0.8, overwrite=False):
    '''
    Plot data distribution and compute K-L divergence for selected features 
    to compare real and synthetic data.
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
                targ_vals_R = data_DF.query(
                    "R_S == 'Real' & AGE_GROUP == @age_group & SEX == @sex"
                )[targ_col]
                targ_vals_S = data_DF.query(
                    "R_S == 'Synthetic' & AGE_GROUP == @age_group & SEX == @sex"
                )[targ_col]
                D = compute_KL_divergence(
                    targ_vals_R, targ_vals_S, num_bins=num_bins
                )
                p_value = test_real_vs_synthetic(
                    targ_vals_R, targ_vals_S, D
                )
                ## Plot:
                fig_idx = ori_idx * 2 + targ_idx + 1
                ax = plt.subplot(len(orientations), 2, fig_idx)
                ax.hist(
                    [targ_vals_R, targ_vals_S], bins=num_bins, density=True, 
                    color=['#FF9933', '#3399FF'], alpha=alpha, 
                    label=[f"R ({len(targ_vals_R)})", f"S ({len(targ_vals_S)})"]
                )
                ax.set_xlabel(targ_col, fontsize=12)
                ax.set_ylabel("")
                ax.set_title(f"K-L Divergence = {D:.2f} (p = {p_value:.2f})", fontsize=14)
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

def save_model_info(result_DF, label_cols, output_path, overwrite=False):
    '''
    Save the model types and feature numbers to a .csv table.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        result_DF["Info"] = result_DF.apply(
            lambda x: f"{x['Model']}_{x['NumberOfFeatures']}", axis=1
        )
        model_info = (result_DF
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
    # - info_DF: DataFrame with unique subjects.
    # - stats_table: Table with median and standard deviation of the data.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        info_DF = (DF
            .loc[:, ["SID", "Age", "Sex", "AgeGroup"]]
            .drop_duplicates("SID")
        )
        stats_table = (info_DF
            .groupby(["Sex", "AgeGroup"])["Age"]
            .agg(["count", "median", "std"])
            .rename(columns = {"count": "N", "median": "Median", "std": "STD"})
            .reset_index()
        )
        stats_table.to_csv(output_path, index=False)
        print(f"\nData median and std are saved to:\n{output_path}")

def modify_DF(result_DF, desc):
    '''
    Modify the DataFrame and transform it to long format.
    <returns>:
    - long_result_DF: DataFrame in long format.
    '''
    result_DF.rename(columns={
        "PredictedAgeDifference": "PAD", 
        "CorrectedPAD": "PAD_ac"
    }, inplace=True)

    long_result_DF = (result_DF
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
    long_result_DF["PAD_abs_value"] = long_result_DF["PAD_value"].abs()

    long_result_DF = long_result_DF.sort_values(by=["Sex", "AgeGroup"], ascending=False)
    if desc.sep_sex:
        long_result_DF["AgeSex"] = long_result_DF["AgeGroup"] + "_" + long_result_DF["Sex"]

    return long_result_DF

def plot_pad_bars(long_result_DF, x_lab, output_path, overwrite=False):
    '''
    Plot the PAD values.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style="whitegrid")
        sns.set_context("talk", font_scale=1.2)
        g = sns.catplot(
            data=long_result_DF, kind="bar",
            x=x_lab, y="PAD_abs_value", hue="PAD_type", col="Type", dodge=True, 
            errorbar="se", palette="dark", alpha=.6, height=6
        )
        g.set_axis_labels("", "PAD Value")
        plt.savefig(output_path)
        print(f"\nBar plot of the PAD values is saved to:\n{output_path}")
        plt.close()

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
    
def plot_cormat(wide_sub_DF, targ_cols, corrwith_cols=None,
                output_path=None, overwrite=False, 
                x_col_names=None, xr=0, y_col_names=None, yr=0, 
                c_bar=False, font_scale=1.1, figsize=(3, 3), dpi=200):
    '''
    Plot the correlation matrix for sub-dataframes.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        if corrwith_cols is None:
            cormat = wide_sub_DF.loc[:, targ_cols].corr()
            annot_mat = pd.DataFrame(index=cormat.index, columns=cormat.columns, dtype=str)
            for t1 in cormat.index:
                for t2 in cormat.columns: 
                    annot_mat.loc[t1, t2] = format_r(wide_sub_DF[t1], wide_sub_DF[t2])
            mask = np.zeros_like(cormat)
            mask[np.triu_indices_from(mask)] = True
        else:
            cormat = pd.DataFrame(index=targ_cols, columns=corrwith_cols, dtype=float)
            annot_mat = cormat.copy(deep=True) 
            annot_mat = annot_mat.astype(str)
            for t1 in cormat.index:
                for t2 in cormat.columns: 
                    sub_df = wide_sub_DF.loc[:, [t1, t2]].dropna()
                    cormat.loc[t1, t2] = sub_df[t1].corr(sub_df[t2])
                    annot_mat.loc[t1, t2] = format_r(sub_df[t1], sub_df[t2])
            mask = None

        sns.set_theme(style='white', font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) 
        sns.heatmap(
            cormat, mask=mask, # square=True, 
            vmin=-1, vmax=1, cmap="RdBu_r", cbar=c_bar, 
            cbar_kws=None if c_bar is False else {"shrink": 0.5, "label": "$r$"}, 
            annot=pd.DataFrame(annot_mat), fmt = "", # annot_kws={"size": 16}, 
            linewidth=.5
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        if x_col_names is not None: 
            ax.set_xticklabels(x_col_names, rotation=xr)
        if y_col_names is not None: 
            ax.set_yticklabels(y_col_names, rotation=xr)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nCorrelation matrix is saved to:\n{output_path}")
        plt.close()

def calc_pairwise_corr(wide_sub_DF_dict, targ_cols, grouping_col, excel_file, overwrite=False, 
                       output_cols=['X', 'Y', 'n', 'r', 'CI95%', 'p-unc', 'p-corr'], 
                       sort_by='p-unc', p_adj='bonf'):
    '''
    Calculate pairwise correlations and save to an .xlsx file.
    <returns>:
    - corr_DF: DataFrame with pairwise correlations.
    '''
    if overwrite and os.path.exists(excel_file):
        os.remove(excel_file)

    pw_corr_list = []
    for group_name, wide_sub_DF in wide_sub_DF_dict.items():
        pw_corr = (
            pg.pairwise_corr(wide_sub_DF.loc[:, targ_cols], padjust=p_adj)
            .sort_values(by=sort_by)[output_cols]
        )
        if (not os.path.exists(excel_file)) or overwrite:
            if not os.path.exists(excel_file):
                pw_corr.to_excel(excel_file, sheet_name=group_name, index=False)
            else:
                with pd.ExcelWriter(excel_file, mode='a') as writer: 
                    pw_corr.to_excel(writer, sheet_name=group_name, index=False)
            print(f"\nPairwise comparisons is saved to:\n{excel_file}")

        pw_corr.insert(0, grouping_col, group_name)
        pw_corr_list.append(pw_corr)
    
    return pd.concat(pw_corr_list, ignore_index=True)

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

def plot_scatter_from_corr(corr_DF, group_name, wide_sub_DF, t1, t2, p_apply, 
                           output_path, overwrite=False, 
                           font_scale=1.2, figsize=(5, 5), dpi=500):
    '''
    Plot the correlation scatter plot for sub-dataframes.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style='whitegrid', font_scale=font_scale)
        fig = plt.figure(num=None, figsize=figsize, dpi=dpi)
        g = sns.JointGrid(
            data=wide_sub_DF, x=t1, y=t2, height=5, ratio=3
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

        group_corr_df = corr_DF.query("AgeSex == @group_name & X == @t1 & Y == @t2")
        if group_corr_df.empty:
            group_corr_df = corr_DF.query("AgeSex == @group_name & X == @t2 & Y == @t1")
            
        N = group_corr_df["n"].iloc[0]
        r = group_corr_df["r"].iloc[0]
        p = group_corr_df[p_apply].iloc[0]
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
    group_pr = feature_DF.groupby("approach")["domain"].value_counts(normalize=True) * 100
    feature_DF["domain_pr"] = feature_DF.apply(
        lambda row: group_pr[row["approach"]][row["domain"]], axis=1
    )
    feature_DF["overall_pr"] = feature_DF.apply(
        lambda row: group_pr[row["approach"]][row["domain"]] * row["approach_pr"], axis=1
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
        
## Main function: ---------------------------------------------------------------------

def main():
    config = Config()
    pad_type = config.pad_type # for correlations 

    desc = load_description(config)
    grouping_col = "AgeSex" if desc.sep_sex else "AgeGroup"
    
    standardized_features = standardized_feature_list()
    domain_approach_mapping = domain_approach_mapping_dict()

    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    ## Load data:
    data_DF, result_DF, selected_features, trainset_info_DF = load_data(
        config, desc, selected_cols=[
            desc.sid_name, "Age", "PredictedAgeDifference", "CorrectedPAD", 
            "Model", "NumberOfFeatures"
        ]
    )

    ## If the data was synthetized, compare real and synthetic data for each group:
    if desc.data_synthetized: 
        
        for label in desc.label_list:
            if desc.sep_sex:
                age_group, sex = label
                group_name = f"{age_group}_{sex}"
            else:
                age_group = label
                group_name = label

            ## Plot data distribution and compute K-L divergence (of the first two features selected based on PCA-based ranking in each feature orientation):
            targ_col_list = plot_data_dist(
                data_DF, age_group, sex, selected_features, 
                os.path.join(config.output_folder, config.data_hist_fn_template.replace("<GroupName>", group_name)), 
                overwrite=args.overwrite
            )

            targ_col_list

            ## Plot correlation matrices between selected features:
            for S_or_R in ["Synthetic", "Real"]: 
                if desc.sep_sex:
                    sub_data_df = data_DF.query(
                        "R_S == @S_or_R & AGE_GROUP == @age_group & SEX == @sex"
                    )
                else:
                    sub_data_df = data_DF.query(
                        "R_S == @S_or_R & AGE_GROUP == @age_group"
                    )
                plot_cormat(
                    wide_sub_DF=sub_data_df, 
                    targ_cols=targ_col_list, 
                    output_path=os.path.join(config.output_folder, config.data_cormat_fn_template.replace("<GroupName>", group_name).replace("<S_or_R>", S_or_R)), 
                    x_col_names=[ f"#{x+1}" for x in range(len(targ_col_list)) ], # use numeric labels on the x-axis
                    y_col_names=rename_cols(targ_col_list), 
                    figsize=(12, 4), 
                    overwrite=args.overwrite
                )

    ## Aggregate model types and feature numbers (save to a table):
    save_model_info(
        result_DF, desc.label_cols,
        os.path.join(config.output_folder, config.model_info_filename), 
        overwrite=args.overwrite
    )

    ## Aggregate medians and STDs of ages for each participant group (save to a table):
    save_discriptive_table(
        result_DF, 
        os.path.join(config.output_folder, config.disc_table_filename), 
        overwrite=args.overwrite
    )    
    if desc.traintest:
        save_discriptive_table(
            trainset_info_DF, 
            os.path.join(config.output_folder, config.disc_table_filename.replace(".csv", " (training set).csv")), 
            overwrite=args.overwrite
        )

    ## Modify DataFrame and transform it to long format:
    long_result_DF = modify_DF(result_DF, desc)
    
    ## Plot PAD bars: 
    if args.ignore_all:
        config.pad_barplot_filename = config.pad_barplot_filename.replace(".png", " (ignore 'All').png")
    plot_pad_bars(
        long_result_DF, grouping_col, 
        os.path.join(config.output_folder, config.pad_barplot_filename), 
        overwrite=args.overwrite
    )

    ## Pivot data for calculating correlations:
    wide_sub_DF_dict = {}

    for label in desc.label_list:
        if desc.sep_sex:
            age_group, sex = label
            group_name = f"{age_group}_{sex}"
            sub_result_df = long_result_DF.query(
                "AgeSex == @group_name & PAD_type == @pad_type"
            ) 
        else:
            group_name = label
            sub_result_df = long_result_DF.query(
                "AgeGroup == @group_name & PAD_type == @pad_type"
            )
        wide_sub_DF = (sub_result_df
            .loc[:, ["SID", "Type", "PAD_value"]]
            .pivot(index="SID", columns="Type", values="PAD_value")
            .reset_index()
            .rename(columns={"index": "SID"})
        )
        wide_sub_DF_dict[group_name] = wide_sub_DF

        ## Plot correlation matrices:
        fn = config.pad_cormat_fn_template.replace("<GroupName>", group_name)
        if args.ignore_all:
            fn = fn.replace(".png", " (ignore 'All').png")
        plot_cormat(
            wide_sub_DF=wide_sub_DF, 
            targ_cols=sorted([ x[:3] for x in desc.feature_orientations ]), 
            output_path=os.path.join(config.output_folder, fn), 
            figsize=(3, 3) if len(desc.feature_orientations) <= 3 else (4, 4),
            overwrite=args.overwrite
        )
        
        ## Plot correlation matrices with standardized features:         
        fn2 = fn.replace("in", "& standardized features in")
        plot_cormat(
            wide_sub_DF=wide_sub_DF.merge(
                data_DF.loc[:, ["SID"]+standardized_features], 
                on="SID", how="left"
            ), 
            targ_cols=standardized_features, 
            corrwith_cols=[ x[:3] for x in desc.feature_orientations ], 
            x_col_names=[ x[:3] for x in desc.feature_orientations ], 
            y_col_names=standardized_features, yr=90, 
            output_path=os.path.join(config.output_folder, fn2), 
            figsize=(8, 6),
            overwrite=args.overwrite
        )

    ## Plot correlation matrices with standardized features for all groups:
    wide_DF = (
        pd.concat(list(wide_sub_DF_dict.values()))
        .merge(data_DF.loc[:, ["SID"]+standardized_features], on="SID", how="left")
    )     
    fn3 = config.pad_cormat_fn_template.replace("in <GroupName>", 
                                                f"& standardized features across all groups (N={len(wide_DF)})") 
    plot_cormat(
        wide_sub_DF=wide_DF, 
        targ_cols=standardized_features, 
        corrwith_cols=[ x[:3] for x in desc.feature_orientations ], 
        x_col_names=[ x[:3] for x in desc.feature_orientations ], 
        y_col_names=standardized_features, yr=90, 
        output_path=os.path.join(config.output_folder, fn3), 
        figsize=(8, 6),
        overwrite=args.overwrite
    )
    # wide_DF.to_csv(os.path.join(config.output_folder, "PAD values with standardized features.csv"), index=False)

    ## Calculate pairwise correlations (save to different sheets in an .xlsx file):
    corr_DF = calc_pairwise_corr(
        wide_sub_DF_dict=wide_sub_DF_dict, 
        targ_cols=[ x[:3] for x in desc.feature_orientations ], 
        grouping_col=grouping_col, 
        excel_file=os.path.join(config.output_folder, config.pad_corr_table_filename), 
        overwrite=args.overwrite
    )

    ## Plot correlations (scatter plots): 
    for group_name, wide_sub_DF in wide_sub_DF_dict.items():
        for t1, t2 in [("BEH", "FUN"), ("BEH", "STR"), ("FUN", "STR")]:
            plot_scatter_from_corr(
                corr_DF, group_name, wide_sub_DF, t1, t2, 
                p_apply="p-corr" if args.p_adjust else "p-unc", 
                output_path=os.path.join(config.output_folder, config.pad_scatter_fn_template.replace("<GroupName>", group_name).replace("<Type1>", t1).replace("<Type2>", t2)), 
                overwrite=args.overwrite
            )

    ## Prepare a smaller, non-duplicatie dataframe:
    model_info_DF = (
        result_DF.loc[:, desc.label_cols + ["Type", "Model", "NumberOfFeatures"]]
        .drop_duplicates()
    )

    ## Make a dataframe with hierachical labels for selected features:
    for ori_name in desc.feature_orientations: 

        ## One dataframe per group:
        feature_DF_dict = {
            group_name: make_feature_DF(
                ori_name, feature_list, domain_approach_mapping
            ) 
            for group_name, feature_list in selected_features[ori_name].items()
        }

        ## Prepare annotation for each subplot:
        subplot_annots = {}
        for label in desc.label_list: 
            if desc.sep_sex:
                age_group, sex = label
                group_name = f"{age_group}_{sex}"
                model_info = model_info_DF.query(
                    "Type == @ori_name & AgeGroup == @age_group & Sex == @sex"
                )
            else:
                group_name = label
                model_info = model_info_DF.query(
                    "Type == @ori_name & AgeGroup == @group_name"
                )
            model_type = model_info['Model'].iloc[0]
            n_features = model_info['NumberOfFeatures'].iloc[0]
            subplot_annots[group_name] = f"{group_name} ({model_type} - {n_features})" 
                
        ## One set of sunburst charts per feature type:
        plot_feature_sunbursts(
            feature_DF_dict=feature_DF_dict, 
            fig_title=f"{ori_name[:3]}", 
            subplot_annots=subplot_annots, 
            output_path=os.path.join(config.output_folder, config.sunburst_fn_template.replace("<FeatureType>", ori_name[:3])),
            overwrite=args.overwrite
        )

## Finally: ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
    print("\nDone!\n")