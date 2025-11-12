#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from utils import basic_Q_features, ST_features

sys.path.append(os.path.join(os.getcwd(), "..", "scripts"))
from gen_derivatives import Constants, Description, load_model_results, force_agesex_cols
from compare_versions import Config, define_arguments, compare_pad_values

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

## ------------------------------------------------------------------------------------------------------------

def load_result_df(custom_bar_x_lab):
    result_DF_list = []

    for version, input_folder in config.input_folders.items():
        desc_json_path = os.path.join(input_folder, "description.json")
        with open(desc_json_path, 'r', errors='ignore') as f:
            desc_json = json.load(f)
        desc = Description(desc_json, const, args)

        config.result_path = os.path.join(input_folder, "results_{}_{}.json")
        result_DF, _, _ = load_model_results(
            config, desc, const, 
            output_path=os.path.join(out_path, f"results_DF_{version}.csv"), 
            overwrite=args.overwrite
        )
        result_DF["Version"] = version
        result_DF["VerType"] = result_DF["Version"] + "_" + result_DF["Type"]

        result_DF.rename(columns={"PredictedAgeDifference": "PAD", "CorrectedPAD": "PADAC",}, inplace=True)
        result_DF["PAD_abs"] = result_DF["PAD"].abs()
        result_DF["PADAC_abs"] = result_DF["PADAC"].abs()

        if args.custom_bar_x_lab is not None:
            data_DF = pd.read_csv(config.raw_data_path)
            data_DF["SID"] = data_DF["BASIC_INFO_ID"].map(lambda x: x.replace("sub-0", ""))
            temp_DF = force_agesex_cols(DF=result_DF, DF2=data_DF)
            temp_DF.rename(columns={custom_bar_x_lab: "BarGroup"}, inplace=True)
            result_DF = pd.merge(
                result_DF, temp_DF.loc[:, ["SID", "BarGroup"]], on="SID", how="left"
            )
                
        if "Sex" not in result_DF.columns:
            result_DF["Sex"] = ""
            result_DF.replace({"AgeGroup": {"le-44": "Y", "ge-45": "O"}}, inplace=True)
            result_DF.replace({"BarGroup": {"le-44": "Y", "ge-45": "O"}}, inplace=True)
        else:
            result_DF.replace({"AgeGroup": {"all": "", "le-44": "Y", "ge-45": "O"}}, inplace=True)
            result_DF.replace({"BarGroup": {"all": "", "le-44": "Y", "ge-45": "O"}}, inplace=True)
        
        result_DF["Group"] = result_DF["AgeGroup"] + result_DF["Sex"]
        result_DF_list.append(result_DF)

    result_DF = pd.concat(result_DF_list, ignore_index=True)
    result_DF.drop_duplicates(inplace=True)

    return desc, result_DF

def compare_versions(result_DF, feature_orientations, version_list):
    stats_DF_list = []

    for pad_type in ["PAD", "PADAC"]:
        for ori_name in feature_orientations:
            for group in ["Y", "O"]:
                for ver_1, ver_2 in itertools.combinations(version_list, 2):
                    stats_results = compare_pad_values(
                        V1_abs=result_DF.query(
                            f"Type == '{ori_name}' & BarGroup == '{group}' & Version == '{ver_1}'"
                        )[f"{pad_type}_abs"], 
                        V2_abs=result_DF.query(
                            f"Type == '{ori_name}' & BarGroup == '{group}' & Version == '{ver_2}'"
                        )[f"{pad_type}_abs"], 
                        independent=True if args.different_participants else False
                    )
                    stats_results.insert(0, "Type", ori_name)
                    stats_results.insert(1, "Group", group)
                    stats_results.insert(2, "PAD_type", pad_type)
                    stats_results.insert(3, "V1", ver_1)
                    stats_results.insert(4, "V2", ver_2)
                    stats_DF_list.append(stats_results)
        
    return pd.concat(stats_DF_list, ignore_index=True)

def compare_modalities(result_DF, feature_orientations, version_list):
    stats_DF_list = []

    for version in version_list:
        for group in ["Y", "O"]:
            for pad_type in ["PAD", "PADAC"]:
                for ori_1, ori_2 in itertools.combinations(feature_orientations, 2):
                    stats_results = compare_pad_values(
                        V1_abs=result_DF.query(
                            f"Version == '{version}' & BarGroup == '{group}' & Type == '{ori_1}'"
                        )[f"{pad_type}_abs"], 
                        V2_abs=result_DF.query(
                            f"Version == '{version}' & BarGroup == '{group}' & Type == '{ori_2}'"
                        )[f"{pad_type}_abs"], 
                        independent=True if args.different_participants else False
                    )
                    stats_results.insert(0, "Version", version)
                    stats_results.insert(1, "Group", group)
                    stats_results.insert(2, "PAD_type", pad_type)
                    stats_results.insert(3, "V1", ori_1)
                    stats_results.insert(4, "V2", ori_2)
                    stats_DF_list.append(stats_results)
                    
    return pd.concat(stats_DF_list, ignore_index=True)

def plot_comparison_bars(result_DF, stats_DF, out_file, fig_type):
    def _version_per_pad():
        g = sns.catplot(
            data=result_DF, x="BarGroup", y="value", col="Type", kind="bar", errorbar="se", sharex=False, 
            hue="Version", hue_order=["Y", "O"], palette=sns.color_palette("Set2", 2), 
            height=5, aspect=.6, alpha=.8, legend=None
        )

        for col_val, ax in g.axes_dict.items():
            for group, x_pos in zip(["Y", "O"], [0, 1]):
                stats_res = stats_DF.query(f"Type == '{col_val}' & Group == '{group}'")
                p_val = stats_res.iloc[0]["P_value"]
                p_sig = stats_res.iloc[0]["P_sig"]
                # p_sig = "< .001 ***" if p_val < .001 else f"{p_val:.3f} {p_sig}"
                y_pos = max([stats_res["V1_mean"].max(), stats_res["V2_mean"].max()]) + 1.5
                
                if p_val < .05:
                    ax.plot( [x_pos-.2, x_pos+.2], [y_pos, y_pos], color="k", lw=1.5)
                    ax.plot( [x_pos-.2, x_pos-.2], [y_pos, y_pos-.3], color="k", lw=1.5)
                    ax.plot( [x_pos+.2, x_pos+.2], [y_pos, y_pos-.3], color="k", lw=1.5)
                    ax.text(x_pos, y_pos-.3, p_sig, 
                            ha="center", va="bottom", fontsize=20, fontdict={"style": "italic"})
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_title(col_val, fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=18)
            
        g.set(ylim=(0, 12.9))
        g.set_xlabels("")
        g.set_ylabels("")
        g.figure.tight_layout()
        g.figure.savefig(out_file)
        print(f"\nFigure saved: {out_file}")
        plt.close()

    def _version_both_pads():
        g = sns.catplot(
            data=result_DF, x="BarGroup", y="value", col="Type", kind="bar", errorbar="se", sharex=False, 
            hue="GroupxPAD", hue_order=["Y_PAD_abs", "Y_PADAC_abs", "O_PAD_abs", "O_PADAC_abs"], 
            palette=["#E1712B", "#FE9A37", "#2A9689", "#36BBA7"], 
            height=5, aspect=.6, alpha=.8, legend=None
        )
        hatches = itertools.cycle(['', '//'])

        for col_val, ax in g.axes_dict.items():
            sub_stats_DF = stats_DF.query(f"Type == '{col_val}'")
            y_pos = max([sub_stats_DF["V1_mean"].max(), sub_stats_DF["V2_mean"].max()]) + .7
            
            for group, x_pos in zip(["Y_PAD", "Y_PADAC", "O_PAD", "O_PADAC"], np.linspace(-.3, .3, 4)):
                stats_res = sub_stats_DF.query(f"GroupxPAD == '{group}'")
                p_val = stats_res.iloc[0]["P_value"]
                p_sig = stats_res.iloc[0]["P_sig"]
                
                if p_val < .05:
                    y_pos += 1 # offset
                    ax.plot( [x_pos, x_pos+1], [y_pos, y_pos], color="k", lw=1.5)
                    ax.plot( [x_pos, x_pos], [y_pos, y_pos-.3], color="k", lw=1.5)
                    ax.plot( [x_pos+1, x_pos+1], [y_pos, y_pos-.3], color="k", lw=1.5)
                    ax.text((x_pos+.5), y_pos-.5, p_sig, 
                            ha="center", va="bottom", fontsize=19, fontdict={"style": "italic"})
                    
            for i, bar in enumerate(ax.patches):
                if i % 2 == 0:
                    hatch = next(hatches)
                bar.set_hatch(hatch)
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.set_title(col_val, fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=18)
            
        g.set(ylim=(0, 17.9))
        g.set_xlabels("")
        g.set_ylabels("")
        g.figure.tight_layout()
        g.figure.savefig(out_file)
        print(f"\nFigure saved: {out_file}")
        plt.close()
    
    def _modalities_per_pad():
        g = sns.catplot(
            data=result_DF, x="Type", y="value", col="VerxGroup", kind="bar", errorbar="se", sharex=False, 
            hue="Type", hue_order=["STR", "BEH", "FUN", "ALL"], # palette=sns.color_palette("husl", 4), 
            palette=[list(sns.color_palette("husl", 5))[0], list(sns.color_palette("husl", 5))[1], list(sns.color_palette("husl", 5))[3], list(sns.color_palette("husl", 5))[4]], 
            height=5, aspect=.6, alpha=.8, legend=None
        )
        x_pos_dict = dict(zip(["STR", "BEH", "FUN", "ALL"], range(4)))

        for col_val, ax in g.axes_dict.items():
            version, group = col_val.split("_")
            sub_stats_DF = stats_DF.query(f"Version == '{version}' & Group == '{group}' & PAD_type == '{pad_type}'")
            y_pos = max([sub_stats_DF["V1_mean"].max(), sub_stats_DF["V2_mean"].max()]) + .5

            for ori_1, ori_2 in itertools.combinations(["STR", "BEH", "FUN", "ALL"], 2):
                stats_res = sub_stats_DF.query(f"V1 == '{ori_1}' & V2 == '{ori_2}'")
                p_val = stats_res.iloc[0]["P_value"]
                p_sig = stats_res.iloc[0]["P_sig"]
                    
                if p_val < .05:
                    x_pos_1 = x_pos_dict[ori_1]
                    x_pos_2 = x_pos_dict[ori_2]
                    y_pos += 1.1 # offset
                    ax.plot( [x_pos_1, x_pos_2], [y_pos, y_pos], color="k", lw=1.5)
                    ax.plot( [x_pos_1, x_pos_1], [y_pos, y_pos-.2], color="k", lw=1.5)
                    ax.plot( [x_pos_2, x_pos_2], [y_pos, y_pos-.2], color="k", lw=1.5)
                    ax.text(np.mean([x_pos_1, x_pos_2]), y_pos-.5, p_sig, 
                            ha="center", va="bottom", fontsize=20, fontdict={"style": "italic"})
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_title(col_val, fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=18)
            
        g.set(ylim=(0, 15.9))
        g.set_xlabels("")
        g.set_ylabels("")
        g.figure.tight_layout()
        g.figure.savefig(out_file)
        print(f"\nFigure saved: {out_file}")
        plt.close()

    def _modalities_both_pads():
        g = sns.catplot(
            data=result_DF, x="var", y="value", col="VerxGroup", kind="bar", errorbar="se", sharex=False, 
            hue="Type", hue_order=["STR", "BEH", "FUN", "ALL"], palette=sns.color_palette("husl", 4), 
            height=5, aspect=.6, alpha=.8, legend=None
        )
        x_pos_dict = {
            "PAD": dict(zip(["STR", "BEH", "FUN", "ALL"], np.linspace(-.3, .3, 4))), 
            "PADAC": dict(zip(["STR", "BEH", "FUN", "ALL"], np.linspace(-.3, .3, 4) + 1))
        }

        for col_val, ax in g.axes_dict.items():
            version, group = col_val.split("_")

            for pad_type in ["PAD", "PADAC"]: 
                sub_stats_DF = stats_DF.query(f"Version == '{version}' & Group == '{group}' & PAD_type == '{pad_type}'")
                y_pos = max([sub_stats_DF["V1_mean"].max(), sub_stats_DF["V2_mean"].max()])

                for ori_1, ori_2 in itertools.combinations(["STR", "BEH", "FUN", "ALL"], 2):
                    stats_res = sub_stats_DF.query(f"V1 == '{ori_1}' & V2 == '{ori_2}'")
                    p_val = stats_res.iloc[0]["P_value"]
                    p_sig = stats_res.iloc[0]["P_sig"]
                    
                    if p_val < .05:
                        x_pos_1 = x_pos_dict[pad_type][ori_1]
                        x_pos_2 = x_pos_dict[pad_type][ori_2]
                        y_pos += 2.5 # offset
                        ax.plot( [x_pos_1, x_pos_2], [y_pos, y_pos], color="k", lw=1.5)
                        ax.plot( [x_pos_1, x_pos_1], [y_pos, y_pos-.3], color="k", lw=1.5)
                        ax.plot( [x_pos_2, x_pos_2], [y_pos, y_pos-.5], color="k", lw=1.5)
                        ax.text(np.mean([x_pos_1, x_pos_2]), y_pos-.5, p_sig, 
                                ha="center", va="bottom", fontsize=20, fontdict={"style": "italic"})

            for i, bar in enumerate(ax.patches):
                if i % 2 == 1:
                    bar.set_hatch('//')
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.set_title(col_val, fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=18)
            
        g.set(ylim=(0, 23))
        g.figure.tight_layout()
        g.set_xlabels("")
        g.set_ylabels("")

    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    if fig_type == "_version_both_pads":
        _version_per_pad()
    elif fig_type == "version_both_pads":
        _version_both_pads()
    elif fig_type == "modalities_per_pad":
        _modalities_per_pad()
    elif fig_type == "modalities_both_pads":
        _modalities_both_pads()

def make_df_for_cormat(result_DF, data_DF):
    DF_Y = result_DF.query("Version == 'ByAge' & AgeGroup == 'Y'")
    DF_O = result_DF.query("Version == 'ByAge' & AgeGroup == 'O'")
    DF_Un = result_DF.query("Version == 'Undivided'")
    data_DF_interested = data_DF.loc[:, ["SID"] + basic_q_features + st_features]

    wide_df_dict = {}
    for pad_type in ["PAD", "PADAC"]:
        wide_df_dict[pad_type] = {}            
        for group_name, sub_df in zip(["Y", "O", "Y&O"], [DF_Y, DF_O, DF_Un]):
            sub_df = (
                sub_df.loc[:, ["Type", "SID", pad_type]]
                .pivot(index="SID", columns="Type", values=pad_type)
                .reset_index()
                .rename(columns={"index": "SID"})
            )
            wide_df_dict[pad_type][group_name] = sub_df.merge(data_DF_interested, on="SID", how="left")

    return wide_df_dict

def plot_corr_mat(DF, targ_cols, out_file):
    targ_df = DF.loc[:, targ_cols]
    cormat = targ_df.corr()
    mask = np.zeros_like(cormat)
    mask[np.triu_indices_from(mask)] = True
    
    sns.set_theme(style='white', font_scale=1.2)
    plt.figure(figsize=(4, 3), dpi=200)
    g = sns.heatmap(
        cormat, mask=mask, # square=True, 
        vmin=-1, vmax=1, linewidth=.5, cmap="RdBu_r", cbar=False, 
        annot=True, fmt = ".2f", annot_kws={"size": 16}, 
    )
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"\nFigure saved: {out_file}")
    plt.close()

def plot_corwith_mat(DF, x_cols, y_cols, out_file, fig_size):
    def _def_feature_names(y_title):
        return {
            "questionnaire": [
                "Edinburgh Handedness Inventory, Sum", 
                "Physical Function", 
                "Physical Limit", 
                "Emotional Well", 
                "Emotional Limit", 
                "Energy", 
                "Social Function", 
                "Pain", 
                "General Health", 
                "Physical", 
                "Mental", 
                "Sleep Quality", 
                "Sleep Latency", 
                "Sleep Duration", 
                "Sleep Efficiency", 
                "Sleep Disturbance", 
                "Sleep Medication", 
                "Daytime Dysfunction", 
                "Sum", 
                "IPAQ, Metabolic Equivalent of Task", 
                "Extraversion", 
                "Agreeableness", 
                "Conscientiousness", 
                "Emotional Stability", 
                "Intellect", 
                "Multidimensional Scale of Perceived Social Support, Sum", 
                "Cognitive Failure Scale, Sum", 
                "Anxiety", 
                "Depression", 
                "Montreal Cognitive Assessment, Sum" 
            ], 
            "standardized test": [
                "Similarity", 
                "Vocabulary", 
                "Information", 
                "Auditory Immediate", 
                "Visual Immediate", 
                "Working Memory", 
                "Logical Memory 1", 
                "Facial Memory 1", 
                "Verbal Pair 1", 
                "Family Picture 1", 
                "Letter Number Sequence", 
                "Spatial Forward", 
                "Spatial Backward", 
                "Fine Motor", 
                "Balance", 
                "Processing Speed" 
            ]
        }[y_title]

    def _create_annot_mat(cormat, p_stacked, fdr=0.05):
        q_vals = fdrcorrection(p_stacked.dropna().values, alpha=fdr)[1]
        q_stacked = pd.Series(q_vals, index=p_stacked.index)
        q_mat = q_stacked.unstack()
        # q_sig = q_mat.map(lambda x: "*" * sum( x <= t for t in [0.05, 0.01, 0.001] ))
        # annot_mat = cormat.map(lambda x: f"{x:.2f}") + q_sig.reindex_like(cormat)
        q_mask = q_mat < .05
        annot_mat = cormat.map(lambda x: f"{x:.2f}") * q_mask.reindex_like(cormat)
        return annot_mat.fillna("--")
    
    cormat = pd.DataFrame(index=y_cols, columns=x_cols, dtype=float)
    p_mat = pd.DataFrame(index=y_cols, columns=x_cols, dtype=str)
    for t1 in y_cols:
        for t2 in x_cols:
            targ_df = DF[[t1, t2]].dropna()
            cormat.loc[t1, t2], p_mat.loc[t1, t2] = pearsonr(targ_df[t1], targ_df[t2])
    p_stacked = p_mat.stack()
    annot_mat = _create_annot_mat(cormat, p_stacked)
    mask = None
    
    sns.set_theme(style='white', font_scale=1.1)
    plt.figure(figsize=fig_size, dpi=200)
    g = sns.heatmap(
        cormat, mask=mask, square=False, 
        vmin=-1, vmax=1, linewidth=.5, cmap="RdBu_r", cbar=False, 
        # cbar=True, cbar_kws={"shrink": 0.5, "label": "$r$"}, 
        annot=pd.DataFrame(annot_mat), fmt = "", annot_kws={"size": 18}, 
        xticklabels=x_cols, yticklabels=_def_feature_names(y_title)
    )
    g.set(xlabel="", ylabel="")
    g.tick_params(axis="both", which="major", labelsize=20)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"\nFigure saved: {out_file}")
    plt.close()

if __name__ == "__main__":
    const = Constants()
    basic_q_features = basic_Q_features()
    st_features = ST_features()

    for model_type in ["ElasticNet", "CART", "RF", "XGBM", "LGBM"]:
        out_path = os.path.join("..", "derivatives", "2025-11-12 Psychonomic poster", model_type)
        os.makedirs(out_path, exist_ok=True)

        ## Define arguments
        sys.argv = [
            "compare_versions.py", 
            "-v", "0", 
            "-ba", f"2025-09-17_original_sex-0_{model_type}", 
            "-un", f"2025-09-17_original_age-0_sex-0_{model_type}", 
            "-cbg", "0"
        ]
        args = define_arguments()
        args.ignore_all = False
        custom_bar_x_lab = "AgeGroup"

        config = Config(args)    
        version_list = list(config.input_folders.keys()) # ["ByAge", "Undivided"]

        ## Load results:
        desc, result_DF = load_result_df(custom_bar_x_lab)
        feature_orientations = result_DF["Type"].unique()

        melted_result_DF = (
            result_DF
            .loc[:, ["Version", "Type", "BarGroup", "SID", "PAD_abs", "PADAC_abs"]]
            .melt(id_vars=["Version", "Type", "BarGroup", "SID"], 
                  value_vars=["PAD_abs", "PADAC_abs"], 
                  var_name="PAD_type")
        )

        ## Bar plots:
        ## Test if an age-stratified approach improves prediction accuracy:
        stats_versions_DF = compare_versions(result_DF, feature_orientations, version_list)
        stats_versions_DF.to_csv(os.path.join(out_path, "compare_versions_stats.csv"), index=False)

        for pad_type in ["PAD", "PADAC"]:
            plot_comparison_bars(
                result_DF=melted_result_DF.query(f"PAD_type == '{pad_type}_abs'"), 
                stats_DF=stats_versions_DF.query(f"PAD_type == '{pad_type}'"), 
                out_file=os.path.join(out_path, f"compare_versions_{pad_type}_bars.png"), 
                fig_type="_version_both_pads"
            )

        # melted_result_DF["GroupxPAD"] = melted_result_DF["BarGroup"].astype(str) + "_" + melted_result_DF["var"]
        # stats_versions_DF["GroupxPAD"] = stats_versions_DF["Group"] + "_" + stats_versions_DF["PAD_type"]
        # plot_comparison_bars( ... , fig_type="version_both_pads")

        ## Compare models trained using features in specific modalities:
        stats_modalities_DF = compare_modalities(result_DF, feature_orientations, version_list)
        stats_modalities_DF.to_csv(os.path.join(out_path, "compare_modalities_stats.csv"), index=False)
        
        melted_result_DF["VerxGroup"] = melted_result_DF["Version"] + "_" + melted_result_DF["BarGroup"].astype(str)

        for pad_type in ["PAD", "PADAC"]:
            plot_comparison_bars(
                result_DF=melted_result_DF.query(f"PAD_type == '{pad_type}_abs'"), 
                stats_DF=stats_modalities_DF.query(f"PAD_type == '{pad_type}'"), 
                out_file=os.path.join(out_path, f"compare_modalities_{pad_type}_bars.png"), 
                fig_type="modalities_per_pad"
            )

        ## Correlation matrices:
        data_DF = pd.read_csv(config.raw_data_path)
        data_DF["SID"] = data_DF["BASIC_INFO_ID"].map(lambda x: x.replace("sub-0", ""))

        wide_df_dict = make_df_for_cormat(result_DF, data_DF)

        for pad_type in ["PAD", "PADAC"]:
            for group_name in ["Y", "O", "Y&O"]:

                ## Among PAC(AC) values:
                plot_corr_mat(
                    DF=wide_df_dict[pad_type][group_name], 
                    targ_cols=sorted(desc.feature_oris), 
                    out_file=os.path.join(out_path, f"cormat_{pad_type}_{group_name}.png")
                )

                ## Correlation with questionnaire / standardized test features:
                for y_title, y_cols, fig_size in zip(
                        ["questionnaire", "standardized test"], 
                        [basic_q_features, st_features], 
                        [(12, 11), (8, 7)]
                    ):
                    plot_corwith_mat(
                        DF=wide_df_dict[pad_type][group_name], 
                        x_cols=sorted(desc.feature_oris), 
                        y_cols=y_cols, 
                        out_file=os.path.join(out_path, f"cormat_{pad_type}_{y_title}_{group_name}.png"), 
                        fig_size=fig_size
                    )
