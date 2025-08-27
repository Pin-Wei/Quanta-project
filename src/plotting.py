#!/usr/bin/python

import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr, entropy 
from statsmodels.stats.multitest import fdrcorrection

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

## scatter plot -----------------------------------------------------------------------

def plot_real_pred_age(DF, y_lab, color_hue, color_dict, output_path, 
                       x_lim=(15, 85), y_lim=(15, 85), 
                       x_ticks=np.arange(20, 85, 10), y_ticks=np.arange(20, 85, 10),
                       font_scale=1.2, fig_size=(5, 5), dpi=500, overwrite=False):
    '''
    Draw the relationship between participants' real ages and predicted ages.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style='whitegrid', font_scale=font_scale)
        plt.figure(figsize=fig_size, dpi=dpi)
        g = sns.scatterplot(
            data=DF, x="Age", y=y_lab, 
            hue=color_hue, palette=color_dict, alpha=.7, 
            legend=False
        )
        g.set(
            xlim=x_lim, ylim=y_lim, xticks=x_ticks, yticks=y_ticks
        )
        plt.plot(
            g.get_xlim(), g.get_ylim(), color="k", linewidth=1, linestyle="--"
        )
        g.set(xlabel="Real age", ylabel="Predicted age")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"\nCorrelation between predicted age and real age is saved to:\n{output_path}")

def plot_corr_with_stats(DF, corr_table, x_lab, y_lab, p_apply, output_path, 
                         font_scale=1.2, fig_size=(5, 5), dpi=500, overwrite=False):
    '''
    Plot the correlation between x and y
    and print the previously calculated statistics (stored in corr_table).
    '''
    def _format_p(p):
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
    
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style='whitegrid', font_scale=font_scale)
        plt.figure(figsize=fig_size, dpi=dpi)
        g = sns.JointGrid(
            data=DF, x=x_lab, y=y_lab, height=5, ratio=3
        )
        g = g.plot_joint(
            sns.regplot, x_jitter=False, y_jitter=False, 
            scatter_kws={'alpha': 0.5, 'edgecolor': 'white'}, line_kws={'linewidth': 1}
        )
        g = g.plot_marginals(
            sns.histplot, kde=True, linewidth=1, bins=15 # binwidth=2
        )
        g.refline(x=0, color='g')
        g.refline(y=0, color='g')
        N = corr_table["n"].iloc[0]
        r = corr_table["r"].iloc[0]
        p = corr_table[p_apply].iloc[0]
        p_print = _format_p(p).replace("p", "p-unc") if p_apply == "p-unc" else _format_p(p)
        plt.suptitle(f"r = {r:.2f}, {p_print}, N = {N:.0f}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"\nCorrelation plot is saved to:\n{output_path}")

## histogram plot ---------------------------------------------------------------------

def plot_real_syn_data(DF, group_name, targeted_features, n_rows, n_cols, 
                       color_dict, output_path, 
                       num_bins=30, alpha=0.8, dpi=200, overwrite=False):
    '''
    Draw the distribution of real and synthetic data and compute the KL divergence
    for targeted features in a given group.
    '''
    def _compute_KL_divergence(real_data, syn_data, num_bins=30):
        '''
        Compute relative entropy (also known as Kullback-Leibler divergence) between two distributions.
        '''
        data_min = min(real_data.min(), syn_data.min())
        data_max = max(real_data.max(), syn_data.max())
        hist_bins = np.linspace(data_min, data_max, num_bins + 1)
        real_hist, _ = np.histogram(real_data, bins=hist_bins)
        synth_hist, _ = np.histogram(syn_data, bins=hist_bins)
        D = entropy(pk=synth_hist + 1e-10, qk=real_hist + 1e-10) # add a small number to avoid division by zero
        return D

    def _test_D_significance(real_data, syn_data, observed_D, n_permutations=1000):
        '''
        Perform a permutation (Monte Carlo) test to statistically assess 
        whether the KL divergence is small enough 
        to conclude that a synthetic dataset matches the original one in distribution. 
        '''
        combined_data = pd.concat([real_data, syn_data], ignore_index=True)
        labels = ['Real'] * len(real_data) + ['Synthetic'] * len(syn_data)
        permuted_D_list = []
        for _ in range(n_permutations):
            shuffled_labels = np.random.permutation(labels)
            G1 = combined_data[np.array(shuffled_labels) == 'Real']
            G2 = combined_data[np.array(shuffled_labels) == 'Synthetic']
            D = _compute_KL_divergence(G1, G2)
            permuted_D_list.append(D)
        p_value = (np.array(permuted_D_list) > observed_D).mean()
        return p_value

    if (not os.path.exists(output_path)) or overwrite:
        plt.style.use('seaborn-v0_8-white')
        fig = plt.figure(figsize=(12, 10), dpi=dpi)
        handles, labels = [], []
        idx = 0
        for nr in range(n_rows):
            for nc in range(n_cols):
                targ_col = targeted_features[idx]
                real_data = DF[DF["R_S"] == "Real"][targ_col]
                syn_data = DF[DF["R_S"] == "Synthetic"][targ_col]
                D = _compute_KL_divergence(real_data, syn_data, num_bins=num_bins)
                p_value = _test_D_significance(real_data, syn_data, D)
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                ax.hist(
                    [real_data, syn_data], bins=num_bins, 
                    density=True, # normalize the histogram
                    color=color_dict.values(), alpha=alpha, 
                    label=[f"R ({len(real_data)})", f"S ({len(syn_data)})"]
                )
                ax.set_xlabel(targ_col, fontsize=12)
                ax.set_ylabel("")
                ax.set_title(f"K-L Divergence = {D:.2f} (p = {p_value:.2f})", fontsize=14)
                handles.append(ax.get_legend_handles_labels()[0])
                labels.append(ax.get_legend_handles_labels()[1])
                idx += 1
        fig.suptitle(group_name.replace("_", ", "), fontsize=18)
        plt.tight_layout()
        plt.figlegend(sum(handles, []), labels, bbox_to_anchor=(1.1, 0.55), fontsize=12)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"\nDistribution plots is saved to:\n{output_path}")

## bar plot ---------------------------------------------------------------------------

def plot_pad_bars(DF, x_lab, one_or_many, color_dict, output_path, 
                  y_lab="PAD_abs_value", z_lab="Type", 
                  type_col="PAD_type", pad_name="PAD", padac_name="PAD_ac",
                  y_max=None, overwrite=False):
    '''
    Draw the PAD values and PAD_ac values for a specific model (one) or participant group (many).
    '''
    if one_or_many == "one":
        mean_pad = DF[DF[type_col] == pad_name][y_lab].mean()
        mean_padac = DF[DF[type_col] == padac_name][y_lab].mean()
        kwargs = {"legend": False}
        y_name = ""
    else:
        kwargs = {"col": z_lab, "col_order": list(DF[z_lab].unique()).sort(key=lambda x: ["STR", "BEH", "FUN", "ALL"].index(x))}
        y_name = "PAD Value"

    if (not os.path.exists(output_path)) or overwrite:
        DF[x_lab] = DF[x_lab].map(lambda x: x.replace("all_", "")) # for better x-axis labels
        sns.set_theme(style="whitegrid")
        sns.set_context("talk", font_scale=1.2)
        g = sns.catplot(
            data=DF, kind="bar", errorbar="se", 
            x=x_lab, y=y_lab, hue=type_col, hue_order=[pad_name, padac_name], 
            palette=color_dict, height=6, aspect=1, alpha=.8, dodge=True, **kwargs
        )
        g.set_axis_labels("", y_name)
        g.set(ylim=(0, y_max)) # for consistency across versions

        if one_or_many == "one":
            g.refline(y=mean_padac, color=color_dict[padac_name], linestyle='--')
            g.refline(y=mean_pad, color=color_dict[pad_name], linestyle='--')
            text = f"mean PAD = {mean_pad:.2f}, PAD_ac = {mean_padac:.2f}"
            g.fig.text(0.5, 0.1, text, ha='center', va='top', fontsize=22, color="#FF006E")
            # g.fig.suptitle(text, fontsize=20)
            ax = g.axes.flat[0]
            ax.tick_params(axis='x', labelsize=20)
            plt.subplots_adjust(bottom=0.2)

        plt.savefig(output_path)
        plt.close()
        print(f"\nBar plot of the PAD values is saved to:\n{output_path}")
        
def plot_categorical_bars(DF, pad_col, version_list, color_dict, 
                          x_lab="Group", col_lab="Version"):
    '''
    Draw the PAD (or PAD_ac) values between different versions.
    '''
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.2)
    g = sns.catplot(
        data=DF, x=x_lab, y=pad_col, kind="bar", errorbar="se", sharex=False, 
        col=col_lab, col_order=version_list,
        hue=col_lab, hue_order=version_list, palette=color_dict, 
        height=5, aspect=.6, alpha=.8, legend=None
    )
    for col_val, ax in g.axes_dict.items():
        pad_mean = DF[DF[col_lab] == col_val][pad_col].mean()
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

def plot_bars_with_stats(result_DF, stats_DF, pad_col, version_list, color_dict, 
                         x_lab="Version", potential_y_lim=None, print_p=False, fig_size=(6, 5)):
    '''
    Draw bar plots and significance markers
    to compare the PAD (or PAD_ac) values between different versions.
    '''
    y_pos = max([stats_DF["V1_mean"].max(), stats_DF["V2_mean"].max()])
    y_offset = max([y_pos, potential_y_lim]) * .2
    p_offset = max([y_pos, potential_y_lim]) * .05
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.2)
    plt.figure(figsize=fig_size)
    g = sns.barplot(
        data=result_DF, x=x_lab, y=pad_col, order=version_list, 
        hue=x_lab, hue_order=version_list, palette=color_dict, 
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

def plot_feature_importances(feature_importances, output_path, 
                             x_lim=(-1, 1), fig_size=(6, 20), overwrite=False):
    '''
    Plot feature importances as horizontal bars.
    '''
    if (not os.path.exists(output_path)) or overwrite:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=fig_size)
        g = sns.barplot(
            x=1, y=0, data=feature_importances
        )
        g.set(xlim=x_lim, ylabel="", xlabel="Feature Importances")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"\nFeature importances plot is saved to:\n{output_path}")

## heatmap ----------------------------------------------------------------------------

def plot_cormat(DF, targ_cols, output_path, fdr=0.05, corrwith_cols=None, 
                x_col_names=None, y_col_names=None, shorter_xcol_names=False, 
                xr=0, yr=0, c_bar=False, square=False, 
                font_scale=1.1, figsize=(3, 3), dpi=200, overwrite=False):
    '''
    Compute correlation matrix and plot it as a heatmap.
    '''
    def _corr_p(x, y):
        mask = ~ (np.isnan(x) | np.isnan(y)) # only use rows without NaN in either column
        if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
            return np.nan # correlation is undefined if either variable has zero variance
        else:
            return pearsonr(x[mask], y[mask])[1]

    def _create_annot_mat(cormat, p_stacked, fdr=fdr):
        q_vals = fdrcorrection(p_stacked.dropna().values, alpha=fdr)[1]
        q_stacked = pd.Series(q_vals, index=p_stacked.index)
        q_mat = q_stacked.unstack()
        q_sig = q_mat.map(lambda x: "*" * sum( x <= t for t in [0.05, 0.01, 0.001] ))
        annot_mat = cormat.map(lambda x: f"{x:.2f}") + q_sig.reindex_like(cormat)
        return annot_mat.fillna("--")
            
    def _rename_labels(col_names):
        renamed_labels = []
        for x, col in enumerate(col_names):
            renamed_labels.append(f"({col}) #{x+1}")
        return renamed_labels

    if (not os.path.exists(output_path)) or overwrite:
        if corrwith_cols is None:
            cormat = DF[targ_cols].corr()
            p_mat = DF[targ_cols].corr(method=lambda x, y: _corr_p(x, y))
            p_stacked = p_mat.where(np.tril(np.ones(p_mat.shape), k=-1).astype(bool)).stack()
            annot_mat = _create_annot_mat(cormat, p_stacked)
            mask = np.zeros_like(cormat)
            mask[np.triu_indices_from(mask)] = True
        else:
            cormat = pd.DataFrame(index=targ_cols, columns=corrwith_cols, dtype=float)
            p_mat = pd.DataFrame(index=targ_cols, columns=corrwith_cols, dtype=str)
            for t1 in targ_cols:
                for t2 in corrwith_cols:
                    targ_df = DF[[t1, t2]].dropna()
                    cormat.loc[t1, t2], p_mat.loc[t1, t2] = pearsonr(targ_df[t1], targ_df[t2])
            p_stacked = p_mat.stack()
            annot_mat = _create_annot_mat(cormat, p_stacked)
            mask = None

        x_col_names = cormat.columns if x_col_names is None else x_col_names
        y_col_names = cormat.index if y_col_names is None else y_col_names

        if shorter_xcol_names:
            if x_col_names == y_col_names:
                x_col_names = [ f"#{x+1}" for x in range(len(x_col_names)) ] # use numeric labels on the x-axis
                y_col_names = _rename_labels(y_col_names) 
            else:
                raise ValueError("x_col_names and y_col_names must be the same if shorter_col_names is True.")

        sns.set_theme(style='white', font_scale=font_scale)
        plt.figure(figsize=figsize, dpi=dpi)
        g = sns.heatmap(
            cormat, mask=mask, square=square, 
            vmin=-1, vmax=1, linewidth=.5, cmap="RdBu_r", cbar=c_bar, 
            cbar_kws=None if c_bar is False else {"shrink": 0.5, "label": "$r$"}, 
            annot=pd.DataFrame(annot_mat), fmt = "", # annot_kws={"size": 16}, 
            xticklabels=x_col_names, yticklabels=y_col_names
        )
        g.set(xlabel="", ylabel="")
        g.set_xticklabels(g.get_xticklabels(), rotation=xr)
        g.set_yticklabels(g.get_yticklabels(), rotation=yr)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"\nCorrelation matrix plot is saved to:\n{output_path}")

## sunburst plot ----------------------------------------------------------------------

def plot_feature_sunburst(feature_DF_dict, parent_col, label_col, color_dict, 
                          one_or_many, fig_title, output_path, 
                          n_cols=None, subplot_annots=None, overwrite=False):
    '''
    Visualize the hierarchical structure of features using a sunburst plot.
    '''
    def _build_sunburst_data(feature_DF, parent_col, label_col, color_dict):
        labels, parents, values, colors, ids = [], [], [], [], []
        approach_counts = feature_DF[parent_col].value_counts().to_dict()
        domain_counts = feature_DF.groupby([parent_col, label_col]).size().to_dict()

        for approach_label, count in approach_counts.items():
            labels.append(approach_label)
            parents.append("") # root
            values.append(count)
            approach_label_clean = approach_label.split("<br>")[0].split(" (")[0]
            colors.append(color_dict.get(approach_label_clean, "#D3D3D3"))  # fallback: light gray
            ids.append(approach_label) 

        for (approach_label, domain_label), count in domain_counts.items():
            labels.append(domain_label)
            parents.append(approach_label)
            values.append(count)
            domain_label_clean = domain_label.split("<br>")[0].split(" (")[0]
            colors.append(color_dict.get(domain_label_clean, "#D3D3D3"))  # fallback: light gray
            ids.append(f"{approach_label}/{domain_label}")

        return labels, parents, values, colors, ids

    def _plot_one_sunburst(feature_DF, parent_col, label_col, color_dict, one_or_many, fig_title=None):
        labels, parents, values, colors, ids = _build_sunburst_data(
            feature_DF, parent_col, label_col, color_dict
        )
        fig = go.Figure(go.Sunburst(
            labels=labels, parents=parents, values=values, 
            marker=dict(colors=colors), ids=ids, branchvalues="total"
        ))
        if one_or_many == "one":
            fig.update_layout(
                title_text=fig_title, title_x=0.5, title_y=1, 
                font=dict(size=20), template="plotly_white", 
                width=300, height=300, margin=dict(t=50, l=0, r=0, b=0)
            )
        else:
            fig.update_layout(width=500, height=500)

        return fig

    if (not os.path.exists(output_path)) or overwrite:
        if one_or_many == "one":
            fig = _plot_one_sunburst(
                feature_DF, parent_col, label_col, color_dict, one_or_many, fig_title
            )
        else:
            n_cols = 2 if n_cols is None else n_cols
            n_figs = len(feature_DF_dict.keys())
            n_rows = (n_figs + n_cols - 1) // n_cols
            subplot_specifications = [[{'type': 'domain'}] * n_cols] * n_rows
            fig = make_subplots(rows=n_rows, cols=n_cols, specs=subplot_specifications)
            annotations = []
            for x, (group_name, feature_DF) in enumerate(feature_DF_dict.items()):
                sunburst_fig = _plot_one_sunburst(
                    feature_DF, parent_col, label_col, color_dict, one_or_many
                )
                r = (x // 2) + 1
                c = (x % 2) + 1
                fig.add_trace(sunburst_fig.data[0], row=r, col=c)
                annotations.append(dict(
                    x=[0.05, 0.95][c-1], y=[0.5, -0.1][r-1], xref="paper", yref="paper", 
                    showarrow=False, font=dict(size=32), text=subplot_annots[group_name]
                ))
            fig.update_layout(
                title_text=fig_title, 
                title_x=0.5, title_y=0.5, font=dict(size=40), 
                template="plotly_white", 
                annotations=annotations, 
                grid=dict(columns=n_cols, rows=n_rows), 
                width=500 * n_cols, height=400 * n_rows, 
                margin=dict(l=0, r=0, t=10, b=100)
            )
            plt.tight_layout()
            
        pio.write_image(fig, output_path, format="png", scale=2)
        plt.close()
        print(f"\nSunburst plot is saved to:\n{output_path}")

## other plots ------------------------------------------------------------------------

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

