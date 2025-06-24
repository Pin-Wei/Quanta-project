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
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from utils import basic_Q_features, ST_features, domain_approach_mapping_dict

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
    
    * Note: some newly added outputs are not included in this discription

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
        self.combined_results_fn     = "[table] combined results.csv"
        self.pad_corr_table_filename = f"[table] pairwise correlations between {self.pad_type}.xlsx"
        self.feature_df_filename     = "[table] selected features.csv"
        ## Figures: 
        self.data_hist_fn_template   = "[hist] distributions of real and synthetic data in <GroupName>.png"
        self.data_cormat_fn_template = "[cormat] between features in <GroupName>'s <S_or_R> data.png"
        self.pred_corr_real_filename = "[scatter] correlation between true and predicted age.png"
        self.pad_barplot_filename    = "[bar] PAD values.png"
        self.pad_cormat_fn_template  = f"[cormat] between {self.pad_type} in <GroupName>.png"
        self.pad_scatter_fn_template = f"[scatter] between {self.pad_type} in <GroupName> (<Type1> × <Type2>).png"
        self.sunburst_fn_template    = "[pie] <FeatureType>.png"

class Constants:
    def __init__(self):
        self.data_synth_methods = [
            "SMOTENC", "CTGAN", "TVAE"
        ]
        self.test_result_cols = [
            "Age", "PredictedAge", "PredictedAgeDifference", "CorrectedPAD", "CorrectedPredictedAge"
        ]
        self.train_result_cols = [
            "TrainingSubjID", "TrainingAge", "TrainingPredAge", "TrainingPAD", "TrainingPADAC", "TrainingCorPredAge"
        ]
        self.model_info_cols = [
            "Model", "NumberOfFeatures"
        ]
        self.pad_bar_y_lims = [
            {"STR": 9, "BEH": 11, "FUN": 14, "ALL": 11}, 
            {"STR": 14, "BEH": 14, "FUN": 14, "ALL": 14}
        ][1]
    
class ColorDicts:
    def __init__(self): 
        self.sex = {
            "M": "#3399FF", 
            "F": "#FF9933"
        }
        self.train_test = {
            "Train": "#FF1493", 
            "Test": "#00BFFF"
        }
        self.real_synth = {
            "Real": "#FF9933", 
            "Synthetic": "#3399FF"
        }
        self.pad_bars = {
            "PAD":    "#219EBC",
            "PAD_ac": "#FB8500" 
        }
        self.sunburst = {
            "STRUCTURE": {
            ## Parents
                "GM":         "#305973",  # Deep slate blue
                "WM":         "#6FAE9E",  # Deep teal
                "NULL":       "#6A4F83",  # Deep muted violet
            ## Children
                "VOLUME":     "#A5D9D4",  # Light teal
                "ThickAvg":   "#B2C8DF",  # Icy blue-gray
                "ThickStd":   "#B7B1D8",  # Muted lavender
                "FA":         "#F1D4B8"   # Soft peach blush
            },
            "BEH": {
            ## Parents
                "MEMORY":     "#2E7D7E",  # Deep teal
                "MOTOR":      "#E66A28",  # Deep orange
                "LANGUAGE":   "#6A4F83",  # Deep muted violet
            ## Children
                "EXCLUSION":  "#7FC5C0",  # Light teal
                "OSPAN":      "#A5D9D4",  # Lighter teal
                "MST":        "#D6ECEA",  # Lightest teal
                "GFORCE":     "#FFA46B",  # Light orange
                "GOFITTS":    "#FFBD85",  # Lighter orange
                "BILPRESS":   "#FFDAB3",  # Lightest orange
                "SPEECHCOMP": "#9D7FBF",  # Light muted violet
                "WORDNAME":   "#C3B2DB"   # Lightest muted violet
            },
            "FUNCTIONAL": {
            ## Parents
                "MRI":        "#3A4F7A",  # Deep indigo blue
                "EEG":        "#6FAE9E",  # Deep teal
            ## Children
                "MEMORY":     "#A5D9D4",  # Light teal
                "MOTOR":      "#FFA46B",  # Light orange
                "LANGUAGE":   "#C3B2DB"   # Light muted violet
            },
            "ALL": {
            ## Parents
                "STR":        "#4E5D6C",  # Deep gray blue
                "MRI":        "#3A4F7A",  # Deep indigo blue
                "EEG":        "#6FAE9E",  # Deep teal
                "BEH":        "#E67E22",  # Deep orange
            ## Children
                "GM":         "#B2C8DF",  # Pale icy blue-gray
                "WM":         "#CBD5C0",  # Soft pale sage green
                "NULL":       "#9AA0B5",  # Slate gray
                "MEMORY":     "#A5C9B3",  # Calm mint green
                "MOTOR":      "#F6A96C",  # Balanced warm orange
                "LANGUAGE":   "#B7B1D8"   # Muted lavender
            }
        }

def load_description(config, const):
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
                [ int(x.split("-")[1]) for x in self.age_group_labels[:-1] ] + 
                [ np.inf ]
            )
            self.label_cols = ["Sex", "AgeGroup"] if self.sep_sex else ["AgeGroup"]
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
            if desc_json["DataBalancingMethod"] in const.data_synth_methods:
                self.data_synthetized = True
            else:
                self.data_synthetized = False

    return Description()

def load_data(config, desc, const, output_path, overwrite=False):
    '''
    Load: 
    - data_DF (pd.DataFrame): The entire dataset used for model training.
    - selected_features (dict): The features selected to train the model.
    - result_DF (pd.DataFrame): Various results of model training, originally in JSON format.
    - combined_results_DF (pd.DataFrame): Combined results of model training across train/test splits.
    and save the result dataframe to file.
    '''
    print("\nLoading data...")

    ## The entire dataset used for model training:
    if desc.data_synthetized:
        data_DF = pd.read_csv(os.path.join(config.input_folder, "prepared_data (marked).csv"))
    else:
        data_DF = pd.read_csv(os.path.join(config.input_folder, "prepared_data.csv"))
    
    data_DF["SID"] = data_DF["ID"].map(lambda x: x.replace("sub-0", ""))
    data_DF.drop(columns=["ID"], inplace=True)

    data_DF["AGE_GROUP"] = pd.cut(data_DF["BASIC_INFO_AGE"], bins=desc.age_breaks, labels=desc.age_group_labels)
    if desc.sep_sex:
        data_DF["SEX"] = data_DF["BASIC_INFO_SEX"].replace({1: "M", 2: "F"})    

    ## The features selected to train the model and the modeling results:
    selected_features = { o: {} for o in desc.feature_orientations }
    main_results_list, train_results_list = [], []

    for label in desc.label_list:
        if desc.sep_sex:
            age_group, sex = label
            group_name = f"{age_group}_{sex}"
        else:
            age_group = label
            group_name = label

        for ori_name in desc.feature_orientations:
            data_path = os.path.join(config.input_folder, f"results_{group_name}_{ori_name}.json")
            
            if os.path.exists(data_path):
                print(os.path.basename(data_path))

                with open(data_path, 'r', errors='ignore') as f:
                    results = json.load(f)

                    selected_features[ori_name][group_name] = results["FeatureNames"]

                    main_results = pd.DataFrame({ 
                        k: v for k, v in results.items() if k in 
                        [desc.sid_name] + const.test_result_cols + const.model_info_cols
                    })
                    main_results.insert(0, "Type", ori_name)
                    main_results.insert(2, "AgeGroup", age_group)
                    if desc.sep_sex:
                        main_results.insert(2, "Sex", sex)
                    main_results_list.append(main_results)

                    if desc.traintest:
                        train_results = pd.DataFrame({
                            k: v for k, v in results.items() if k in const.train_result_cols
                        })
                        train_results.insert(0, "Type", ori_name)
                        train_results.insert(2, "AgeGroup", age_group)
                        if desc.sep_sex:
                            train_results.insert(2, "Sex", sex)
                        train_results_list.append(train_results)

    result_DF = pd.concat(main_results_list, ignore_index=True)
    result_DF.insert(1, "SID", result_DF[desc.sid_name].map(lambda x: x.replace("sub-0", "")))
    result_DF.drop(columns=[desc.sid_name], inplace=True)

    if desc.traintest: 
        train_results_DF = pd.concat(train_results_list, ignore_index=True)
        train_results_DF.insert(1, "SID", train_results_DF["TrainingSubjID"].map(lambda x: x.replace("sub-0", "")))
        train_results_DF.drop(columns=["TrainingSubjID"], inplace=True)
        train_results_DF.rename(columns={
            "TrainingAge"       : "Age", 
            "TrainingPredAge"   : "PredictedAge", 
            "TrainingPAD"       : "PredictedAgeDifference", 
            "TrainingPADAC"     : "CorrectedPAD", 
            "TrainingCorPredAge": "CorrectedPredictedAge"
        }, inplace=True)
        test_results_DF = result_DF.copy(deep=True)
        test_results_DF.insert(1, "TrainTest", "Test")
        test_results_DF.drop(columns=["Model", "NumberOfFeatures"], inplace=True)
        train_results_DF.insert(1, "TrainTest", "Train")
        combined_results_DF = pd.concat([train_results_DF, test_results_DF])
        combined_results_DF.sort_values(by=["Type"] + desc.label_cols + ["TrainTest"], ascending=False)
    else:
        combined_results_DF = result_DF

    if not os.path.exists(output_path) or overwrite:
        combined_results_DF.to_csv(output_path, index=False)
        print(f"\nResults are saved to:\n{output_path}")

    return data_DF, selected_features, result_DF, combined_results_DF

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

def plot_age_pred_corr(combined_results_DF, y_lab, color_hue, color_dict, 
                       output_path, overwrite=False, 
                       font_scale=1.2, fig_size=(5, 5), dpi=500):
    '''
    Plot the correlation between participants' real ages and predicted ages.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style='whitegrid', font_scale=font_scale)
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        g = sns.scatterplot(
            data=combined_results_DF, x="Age", y=y_lab, 
            hue=color_hue, palette=color_dict, 
            legend=False
        )
        plt.plot(
            g.get_xlim(), g.get_ylim(), color="k", linewidth=1, linestyle="--"
        )
        r, p = pearsonr(combined_results_DF["Age"], combined_results_DF[y_lab])
        p_print = format_p(p)
        N = len(combined_results_DF)        
        g.set_title(f"r = {r:.2f}, {p_print}, N = {N:.0f}")
        g.set_xlabel("Real age")
        g.set_ylabel("Predicted age")
        # g.get_legend().set_title("")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"\nCorrelation between predicted age and real age is saved to:\n{output_path}")

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

def plot_data_dist(data_DF, age_group, sex, selected_features, color_dict, output_path, 
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
                    color=color_dict.values(), alpha=alpha, 
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

def save_model_info(result_DF, label_cols, desc, output_path, overwrite=False):
    '''
    Save the model types and feature numbers to a .csv table.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        result_DF["Info"] = result_DF.apply(
            lambda x: f"{x['Model']} ({x['NumberOfFeatures']})", axis=1
        )
        (result_DF
            .loc[:, label_cols + ["Type", "Info"]]
            .drop_duplicates() 
            .pivot(index = label_cols, 
                   columns = "Type", 
                   values = "Info")
            .loc[:, desc.feature_orientations] # re-order
            .iloc[::-1] # reverse rows
            .to_csv(output_path)
        )
        print(f"\nModel types and feature numbers are saved to:\n{output_path}")

def save_discriptive_table(DF, label_cols, output_path, overwrite=False):
    '''
    Save the median and standard deviation of the data to a .csv table.
    # <previous returns>:
    # - info_DF: DataFrame with unique subjects.
    # - stats_table: Table with median and standard deviation of the data.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        (DF
            .loc[:, ["SID", "Age"] + label_cols]
            .drop_duplicates("SID")
            .groupby(label_cols)["Age"]
            .agg(["count", "median", "std"])
            .rename(columns = {"count": "N", "median": "Median", "std": "STD"})
            .reset_index()
            .to_csv(output_path, index=False)
        )
        print(f"\nData median and std are saved to:\n{output_path}")

def modify_DF(result_DF, desc):
    '''
    Modify the DataFrame, transform it to long format, and save to a .csv file.
    <returns>:
    - long_result_DF: DataFrame in long format.
    '''
    long_result_DF = (
        result_DF
        .rename(columns={
            "PredictedAgeDifference": "PAD", 
            "CorrectedPAD": "PAD_ac"
        })
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
        .sort_values(
            by=desc.label_cols, 
            ascending=False
        )
    )
    long_result_DF["PAD_abs_value"] = long_result_DF["PAD_value"].abs()

    if desc.sep_sex:
        long_result_DF["AgeSex"] = long_result_DF["AgeGroup"] + "_" + long_result_DF["Sex"]

    return long_result_DF

def plot_pad_bar(long_result_DF, x_lab, color_dict, output_path, 
                 y_lim=None, overwrite=False):
    '''
    Plot the PAD values as a bar plot.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style="whitegrid")
        sns.set_context("talk", font_scale=1.2)
        df = long_result_DF.copy(deep=True) # avoid modifying the original DataFrame
        df[x_lab] = df[x_lab].map(lambda x: x.replace("all_", "")) # for better x-axis labels
        mean_pad = df[df["PAD_type"] == "PAD"]["PAD_abs_value"].mean()
        mean_padac = df[df["PAD_type"] == "PAD_ac"]["PAD_abs_value"].mean()
        g = sns.catplot(
            data=df, kind="bar", errorbar="se", 
            x=x_lab, y="PAD_abs_value", hue="PAD_type", hue_order=["PAD", "PAD_ac"], 
            palette=color_dict, height=6, aspect=1, alpha=.8, dodge=True, legend=False
        )
        g.refline(
            y=mean_padac, color=color_dict["PAD_ac"], linestyle='--'
        )
        g.refline(
            y=mean_pad, color=color_dict["PAD"], linestyle='--'
        )
        g.set_axis_labels("", "")
        g.set(ylim=(0, y_lim)) # for consistency across versions
        # g.fig.suptitle(
        #     f"mean PAD = {mean_pad:.2f}, PAD_ac = {mean_padac:.2f}", fontsize=20
        # )
        g.fig.text(
            0.5, 0.1, 
            f"mean PAD = {mean_pad:.2f}, PAD_ac = {mean_padac:.2f}", 
            ha='center', va='top', fontsize=22, color="#FF006E"
        )
        ax = g.axes.flat[0]
        ax.tick_params(axis='x', labelsize=20)
        plt.subplots_adjust(bottom=0.2)
        # plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nBar plot of the PAD values is saved to:\n{output_path}")
        plt.close()

def plot_pad_bars(long_result_DF, x_lab, color_dict, output_path, 
                  y_lim=None, overwrite=False):
    '''
    Plot the PAD values.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style="whitegrid")
        sns.set_context("talk", font_scale=1.2)
        df = long_result_DF.copy(deep=True) # avoid modifying the original DataFrame
        df[x_lab] = df[x_lab].map(lambda x: x.replace("all_", "")) # for better x-axis labels
        g = sns.catplot(
            data=df, kind="bar", errorbar="se", 
            x=x_lab, y="PAD_abs_value", hue="PAD_type", col="Type", 
            hue_order=["PAD", "PAD_ac"], 
            col_order=list(df["Type"].unique()).sort(key=lambda x: ["STR", "BEH", "FUN", "ALL"].index(x)), 
            palette=color_dict, height=6, aspect=1, dodge=True
        )
        g.set_axis_labels("", "PAD Value")
        g.set(ylim=(0, y_lim)) 
        plt.savefig(output_path)
        print(f"\nBar plot of the PAD values is saved to:\n{output_path}")
        plt.close()

def rename_cols(col_list):
    renamed_col_list = []
    for x, col in enumerate(col_list):
        renamed_col_list.append(f"({col}) #{x+1}")
    return renamed_col_list

def format_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return "N/A"
    else:
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
    kwargs = {}
    if x_col_names is not None: 
        kwargs.update({"xticklabels": x_col_names})
    if y_col_names is not None: 
        kwargs.update({"yticklabels": y_col_names})

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
            linewidth=.5, **kwargs
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xr)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=yr)
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

def plot_scatter_from_corr(corr_DF, group_name, wide_sub_DF, t1, t2, grouping_col, p_apply, 
                           output_path, overwrite=False, 
                           font_scale=1.2, fig_size=(5, 5), dpi=500):
    '''
    Plot the correlation scatter plot for sub-dataframes.
    <no returns>
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style='whitegrid', font_scale=font_scale)
        fig = plt.figure(figsize=fig_size, dpi=dpi)
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

        corr_DF = corr_DF[corr_DF[grouping_col] == group_name]
        group_corr_df = corr_DF.query("X == @t1 & Y == @t2")
        if group_corr_df.empty:
            group_corr_df = corr_DF.query("X == @t2 & Y == @t1")
            
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
    dict_list = []
    for feature in feature_list:

        if "MRI" in feature:
            dat_type, task, method, cond, hemi, region, var_name = feature.split("_")
            
            if dat_type == "STRUCTURE":
                if ori_name == "ALL": 
                    dict_list.append({
                        "domain": cond, # "GM", "WM", or "NULL", 
                        "approach": "STR", 
                        "feature": feature
                    })
                else: 
                    dict_list.append({
                        "domain": var_name, # "VOLUME", "ThickAvg", or "FA"
                        "approach": cond, # "GM", "WM", or "NULL"
                        "feature": feature
                    })
            else: # "FUNCTIONAL"
                dict_list.append({
                    "domain": dat_type, # "MEMORY", "MOTOR", or "LANGUAGE"
                    "approach": "MRI", 
                    # "task": task, # "SPEECHCOMP", "WORDNAME", "MST", or "GFORCE"
                    "feature": feature
                })

        elif "EEG" in feature:
            dat_type, task, method, cond, hemi, region, time_freq_var = feature.split("_")[:7]
            
            dict_list.append({
                "domain": dat_type, # "MEMORY", "MOTOR", or "LANGUAGE"
                "approach": "EEG", 
                # "task": task, # "EXCLUSION", "OSPAN", "GOFITTS", or "BILPRESS"
                "feature": feature
            })

        elif "BEH" in feature: 
            dat_type, task, method, cond, var_name = feature.split("_")

            if ori_name == "ALL": 
                dict_list.append({
                    "domain": dat_type, # "MEMORY", "MOTOR", or "LANGUAGE"
                    "approach": "BEH", 
                    "feature": feature
                })
            else: 
                dict_list.append({
                    "domain": task, 
                    "approach": dat_type, # "MEMORY", "MOTOR", or "LANGUAGE"
                    "feature": feature
                })

    feature_DF = pd.DataFrame(dict_list)
    approach_num = feature_DF["approach"].value_counts()
    approach_pr = feature_DF["approach"].value_counts(normalize=True) * 100
    within_approach_domain_num = feature_DF.groupby("approach")["domain"].value_counts()   
    within_approach_domain_pr = feature_DF.groupby("approach")["domain"].value_counts(normalize=True) * 100
    feature_DF["approach_num"] = feature_DF["approach"].map(approach_num)
    feature_DF["approach_and_num"] = feature_DF.apply(
        lambda x: f"{x['approach']} ({x['approach_num']})", axis=1)
    feature_DF["approach_pr"] = feature_DF["approach"].map(approach_pr)
    feature_DF["approach_and_pr"] = feature_DF.apply(
        lambda x: f"{x['approach']}<br>({x['approach_pr']:.1f}%)" if x['approach_pr'] > 7 
        else f"{x['approach']} ({x['approach_pr']:.1f}%)", axis=1)
    feature_DF["domain_num"] = feature_DF.apply(
        lambda x: within_approach_domain_num[x["approach"]][x["domain"]], axis=1)
    feature_DF["domain_and_num"] = feature_DF.apply(
        lambda x: f"{x['domain']} ({x['domain_num']})", axis=1)  
    feature_DF["domain_pr"] = feature_DF.apply(
        lambda x: within_approach_domain_pr[x["approach"]][x["domain"]], axis=1)
    feature_DF["overall_pr"] = feature_DF.apply(
        lambda x: within_approach_domain_pr[x["approach"]][x["domain"]] * x["approach_pr"], axis=1)
    feature_DF["domain_and_pr"] = feature_DF.apply(
        lambda x: f"{x['domain']}<br>({x['domain_pr']:.1f}%)" if x['overall_pr'] > 300 
        else f"{x['domain']} ({x['domain_pr']:.1f}%)", axis=1)
    
    return feature_DF

def build_sunburst_data(feature_df, color_dict, parent_col, label_col):
    '''
    Build the data for the sunburst plot from the feature DataFrame.
    '''
    labels, parents, values, colors, ids = [], [], [], [], []
    approach_counts = (
        feature_df[parent_col].value_counts().to_dict()
    )
    domain_counts = (
        feature_df.groupby([parent_col, label_col]).size().to_dict()
    )
    for approach_label, count in approach_counts.items():
        labels.append(approach_label)
        parents.append("") # root
        values.append(count)
        ids.append(approach_label) 
        approach_label_clean = approach_label.split("<br>")[0].split(" (")[0]
        colors.append(color_dict.get(approach_label_clean, "#D3D3D3"))  # fallback: light gray

    for (approach_label, domain_label), count in domain_counts.items():
        labels.append(domain_label)
        parents.append(approach_label)
        values.append(count)
        ids.append(f"{approach_label}/{domain_label}")
        domain_label_clean = domain_label.split("<br>")[0].split(" (")[0]
        colors.append(color_dict.get(domain_label_clean, "#D3D3D3"))  # fallback: light gray

    return labels, parents, values, colors, ids

def plot_feature_sunburst(feature_DF, color_dict, fig_title, output_path, num=False, overwrite=False):
    '''
    Plot the sunburst plot for the features in the given DataFrame.
    '''
    if num:
        output_path = output_path.replace(".png", " (num).png")
    if (not os.path.exists(output_path)) or overwrite:
        if num:
            labels, parents, values, colors, ids = build_sunburst_data(
                feature_DF, color_dict, "approach_and_num", "domain_and_num"
            )
        else:
            labels, parents, values, colors, ids = build_sunburst_data(
                feature_DF, color_dict, "approach_and_pr", "domain_and_pr"
            )
        fig = go.Figure(go.Sunburst(
            labels=labels, 
            parents=parents, 
            values=values, 
            marker=dict(colors=colors), 
            ids=ids, 
            branchvalues="total"
        ))
        fig.update_layout(
            title_text=fig_title, 
            title_x=0.5, title_y=1, font=dict(size=20), 
            template="plotly_white", 
            width=300, height=300, 
            margin=dict(t=50, l=0, r=0, b=0)
        )
        pio.write_image(fig, output_path, format="png", scale=2)
        plt.close()
        print(f"\nSunburst plot is saved to:\n{output_path}")

def plot_many_feature_sunbursts(feature_DF_dict, color_dict, fig_title, subplot_annots, output_path, 
                                num=False, overwrite=False, ncol=None, nrow=None):
    '''
    Plot the sunburst plots for each dataframe in the dictionary, 
    whose keys are the group names.
    '''
    if num:
        output_path = output_path.replace(".png", " (num).png")
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
            if num:
                labels, parents, values, colors, ids = build_sunburst_data(
                    feature_DF, color_dict, "approach_and_num", "domain_and_num"
                )
            else:
                labels, parents, values, colors, ids = build_sunburst_data(
                    feature_DF, color_dict, "approach_and_pr", "domain_and_pr"
                )
            sunburst_fig = go.Figure(go.Sunburst(
                labels=labels, 
                parents=parents, 
                values=values, 
                marker=dict(colors=colors), 
                ids=ids, 
                branchvalues="total"
            ))
            sunburst_fig.update_layout(width=500, height=500)
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

def plot_color_legend(color_dict, output_path, 
                      fig_size=(4.5, 1.5), n_cols=3, box_size=0.3, overwrite=False):
    '''
    Plot a color legend (default for the feature sunburst plots).
    '''
    n_rows = (len(color_dict) + n_cols - 1) // n_cols

    if (not os.path.exists(output_path)) or overwrite:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.axis("off")
        for idx, (label, color) in enumerate(color_dict.items()):
            x = (idx % n_cols) * 3 # 3 is the horizontal spacing unit
            y = (idx // n_cols) * -1 # make it stacked from the top to the bottom
            rect = mpatches.Rectangle(
                (x, y), width=box_size, height=box_size, color=color
            )
            ax.add_patch(rect)
            ax.text(
                s=label, 
                x=(x + box_size + 0.2), # 0.2 units to the right of the color block
                y=(y + box_size / 2), # half the height of the color block
                ha='left', va='center', fontsize=11
            )
        ax.set_xlim(-0.5, n_cols * 3) # start from -0.5 to make the first color block not touch the edge
        ax.set_ylim(-n_rows, 1) # set 1 as the upper bound to leave space at the top
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"\nColor legend is saved to:\n{output_path}")

## Main function: ---------------------------------------------------------------------

def main():
    config = Config()
    const = Constants()
    desc = load_description(config, const)
    color_dicts = ColorDicts()

    basic_q_features = basic_Q_features()
    st_features = ST_features()
    domain_approach_mapping = domain_approach_mapping_dict()

    pad_type = config.pad_type # for correlations 
    grouping_col = "AgeSex" if desc.sep_sex else "AgeGroup"

    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    ## Load data and save the combined result table:
    data_DF, selected_features, result_DF, combined_results_DF = load_data(
        config, desc, const, 
        output_path=os.path.join(config.output_folder, config.combined_results_fn), 
        overwrite=args.overwrite
    )

    ## Plot correlation between real and predicted ages:
    for ori_name in desc.feature_orientations: 
        fp = os.path.join(config.output_folder, config.pred_corr_real_filename.replace(".png", f" ({ori_name[:3]}).png"))
        
        for file_path, y_lab in [
            (fp, "PredictedAge"), 
            (fp.replace("predicted", "corrected predicted"), "CorrectedPredictedAge")
        ]:
            plot_age_pred_corr(
                combined_results_DF=combined_results_DF.query("Type == @ori_name"), 
                y_lab=y_lab, 
                color_hue=["Sex", "TrainTest"][1], 
                color_dict=[color_dicts.sex, color_dicts.train_test][1], 
                output_path=file_path,
                overwrite=args.overwrite
            )
    plot_color_legend(
        color_dict=color_dicts.train_test, 
        output_path=os.path.join(config.output_folder, "[color legend] Train & Test.png"),
        fig_size=(2, .5), n_cols=2, box_size=0.5, overwrite=args.overwrite
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
                data_DF, age_group, sex, selected_features, color_dicts.real_synth, 
                os.path.join(config.output_folder, config.data_hist_fn_template.replace("<GroupName>", group_name)), 
                overwrite=args.overwrite
            )

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
        result_DF, desc.label_cols, desc, 
        os.path.join(config.output_folder, config.model_info_filename), 
        overwrite=args.overwrite
    )

    ## Aggregate medians and STDs of ages for each participant group (save to a table):
    save_discriptive_table(
        DF=result_DF, 
        label_cols=desc.label_cols,
        output_path=os.path.join(config.output_folder, config.disc_table_filename), 
        overwrite=args.overwrite
    )    
    if desc.traintest:
        save_discriptive_table(
            DF=combined_results_DF.query("TrainTest == 'Train'"), 
            label_cols=desc.label_cols,
            output_path=os.path.join(config.output_folder, config.disc_table_filename.replace(".csv", " (training set).csv")), 
            overwrite=args.overwrite
        )

    ## Modify DataFrame and transform it to long format:
    long_result_DF = modify_DF(result_DF, desc)
    
    ## Plot PAD bars: 
    replace_to = " (ignore 'All').png" if args.ignore_all else ".png"
    plot_pad_bars(
        long_result_DF=long_result_DF, 
        x_lab=grouping_col, 
        color_dict = color_dicts.pad_bars, 
        output_path=os.path.join(config.output_folder, config.pad_barplot_filename.replace(".png", replace_to)), 
        overwrite=args.overwrite
    )
    
    for ori_name in desc.feature_orientations: 
        ori_name= ori_name[:3]
        replace_to = f" ({ori_name}).png"
        plot_pad_bar(
            long_result_DF=long_result_DF.query("Type == @ori_name"), 
            x_lab=grouping_col, 
            color_dict = color_dicts.pad_bars, 
            output_path=os.path.join(config.output_folder, config.pad_barplot_filename.replace(".png", replace_to)), 
            y_lim=const.pad_bar_y_lims[ori_name],
            overwrite=args.overwrite
        )
        
    plot_color_legend(
        color_dict=color_dicts.pad_bars, 
        # color_dict=dict(zip(["PAD", "PAD_ac"], sns.color_dicts("dark", 2))), 
        output_path=os.path.join(config.output_folder, "[color legend] PAD & PAD_ac.png"),
        fig_size=(2, .5), n_cols=2, box_size=0.5, overwrite=args.overwrite
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
        
        ## Plot correlation matrices with standardized/questionnaire features:
        wide_DF = wide_sub_DF.merge(
            data_DF.loc[:, ["SID"]+basic_q_features+st_features], 
            on="SID", how="left"
        )
        plot_cormat( # standardized
            wide_sub_DF=wide_DF, 
            targ_cols=st_features, 
            corrwith_cols=[ x[:3] for x in desc.feature_orientations ], 
            x_col_names=[ x[:3] for x in desc.feature_orientations ], 
            y_col_names=st_features, 
            output_path=os.path.join(config.output_folder, fn.replace("in", f"& standardized features in").replace(".png", f" (N={len(wide_DF)}).png")), 
            figsize=(8, 6),
            overwrite=args.overwrite
        )
        plot_cormat( # questionnaire
            wide_sub_DF=wide_DF, 
            targ_cols=basic_q_features, 
            corrwith_cols=[ x[:3] for x in desc.feature_orientations ], 
            x_col_names=[ x[:3] for x in desc.feature_orientations ], 
            y_col_names=basic_q_features, 
            output_path=os.path.join(config.output_folder, fn.replace("in", f"& questionnaire features in").replace(".png", f" (N={len(wide_DF)}).png")), 
            figsize=(8, 10),
            overwrite=args.overwrite
        )

    ## Plot correlation matrices with standardized features for all groups:
    wide_DF = (
        pd.concat(list(wide_sub_DF_dict.values()))
        .merge(data_DF.loc[:, ["SID"]+st_features], on="SID", how="left")
    )
    # wide_DF.to_csv(os.path.join(config.output_folder, "PAD values with standardized features.csv"), index=False)
    plot_cormat(
        wide_sub_DF=wide_DF, 
        targ_cols=st_features, 
        corrwith_cols=[ x[:3] for x in desc.feature_orientations ], 
        x_col_names=[ x[:3] for x in desc.feature_orientations ], 
        y_col_names=st_features, 
        output_path=os.path.join(config.output_folder, config.pad_cormat_fn_template.replace("in <GroupName>", f"& standardized features across all groups (N={len(wide_DF)})")), 
        figsize=(8, 6),
        overwrite=args.overwrite
    )

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
                corr_DF, group_name, wide_sub_DF, t1, t2, grouping_col, 
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
    feature_DF_list = [] # to concat and save to .csv

    for ori_name in desc.feature_orientations: 

        ## One dataframe per group:
        feature_DF_dict = {
            group_name: make_feature_DF(
                ori_name, feature_list, domain_approach_mapping
            ) 
            for group_name, feature_list in selected_features[ori_name].items()
        }

        for group_name, feature_DF in feature_DF_dict.items(): 
            feature_df = pd.DataFrame(feature_DF)
            feature_df.insert(0, "Type", ori_name[:3])
            feature_df.insert(1, "Group", group_name)
            feature_DF_list.append(feature_df)

        ## Prepare annotation for each subplot:
        subplot_annots, fig_titles = {}, {}
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
            fig_titles[group_name] = f"{model_type} ({n_features})"

        ## One set of sunburst charts per feature type:
        fp = os.path.join(config.output_folder, config.sunburst_fn_template.replace("<FeatureType>", ori_name[:3]))
        plot_many_feature_sunbursts(
            feature_DF_dict=feature_DF_dict, 
            color_dict=color_dicts.sunburst[ori_name], 
            fig_title=f"{ori_name[:3]}", 
            subplot_annots=subplot_annots, 
            output_path=fp, 
            num=True, 
            overwrite=args.overwrite
        )
        
        ## One sunburst chart per group:
        for group_name, feature_DF in feature_DF_dict.items():
            plot_feature_sunburst(
                feature_DF=feature_DF, 
                color_dict=color_dicts.sunburst[ori_name], 
                fig_title=fig_titles[group_name], 
                output_path=fp.replace(".png", f" ({group_name}).png"), 
                num=True, 
                overwrite=args.overwrite
            )

    ## Save the feature dataframe:
    output_path = os.path.join(config.output_folder, config.feature_df_filename)
    if not os.path.exists(os.path.dirname(output_path)) or args.overwrite:
        feature_DF_long = pd.concat(feature_DF_list)
        feature_DF_long.rename(columns={
            "feature"     : "Feature", 
            "domain"      : "Level_2", 
            "domain_num"  : "L2_num", 
            "approach"    : "Level_1", 
            "approach_num": "L1_num"
        }, inplace=True)
        feature_DF_long = feature_DF_long.loc[:, [
            "Type", "Group", "Level_1", "L1_num", "Level_2", "L2_num" # , "Feature"
        ]]
        feature_DF_long.drop_duplicates(inplace=True)
        feature_DF_long.to_csv(output_path, index=False)
        print(f"\nFeature dataframe saved to:\n{output_path}")

## Finally: ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
    print("\nDone!\n")