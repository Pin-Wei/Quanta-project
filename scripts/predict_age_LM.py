#!/usr/bin/env python

# Usage: python predict_age_LM.py [-d <domain>] [-at <age_threshold>] [-ti <terms_included>] [-s <seed>] [-o

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm # https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html

from predict_age_ML import convert_np_types

sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from utils import platform_features

class Config:
    def __init__(self, args):
        self.terms_included = ["linear-only", "with-interaction"][args.terms_included]
        self.root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.raw_data_path = os.path.join(self.root_dir, "data", "rawdata", "DATA_ses-01_2025-05-29.csv")
        self.output_dir = os.path.join(self.root_dir, "outputs", f"{datetime.today().strftime('%Y-%m-%d')}_platform_{self.terms_included}")
        if not getattr(args, "overwrite", False):
            while os.path.exists(self.output_dir):
                self.output_dir += "+"
        os.makedirs(self.output_dir, exist_ok=True)
        self.config_path = os.path.join(self.output_dir, "config.json")
        self.data_path = os.path.join(self.output_dir, "data.csv")
        self.loocv_coefs_path = os.path.join(self.output_dir, "loocv_coefs.csv")
        self.loocv_performance_path = os.path.join(self.output_dir, "loocv_performance.json")
        self.model_coefs_path = os.path.join(self.output_dir, "model_coefs.csv")
        self.model_summ_json_path = os.path.join(self.output_dir, "model_summary.json")
        self.model_summ_txt_path = os.path.join(self.output_dir, "model_summary.txt")

def define_arguments():
    parser = argparse.ArgumentParser(description='''
        This script is for running linear regression models for predicting age from the platform features
        in elderly participants only. The features can be selected from different cognitive domains (Motor, Memory, and/or Language).
    ''')
    parser.add_argument("-d", "--domain", type=int, nargs="+", default=[0], 
                        help="The domain(s) of features to be used for model training. 0: Motor, 1: Memory, 2: Language." + 
                             "Multiple domains can be specified separated by space. Default is 0 (Motor).")
    parser.add_argument("-at", "--age_threshold", type=int, default=60, 
                        help="Age threshold to define elderly group. Default is 60.")
    parser.add_argument("-ti", "--terms_included", type=int, default=1, choices=[0, 1], 
                        help="Terms to be included in the regression model." + 
                             "0: linear terms only; 1: linear and two-way interaction terms. Default is 1.")
    parser.add_argument("-s", "--seed", type=int, default=None, 
                        help="Random seed for reproducibility. If not specified, a random seed will be generated.")
    parser.add_argument("-o", "--overwrite", action="store_true", default=False, 
                        help="Overwrite the output folder if it already exists.")
    return parser.parse_args()

def save_config(config):
    config_dict = {
        k.upper(): v for k, v in config.__dict__.items() 
        if not any(s in k for s in ["path", "dir"]) 
    }
    config_dict = convert_np_types(config_dict)
    with open(config.config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False)

def prepare_data(config, args):
    '''
    Load, select elderly participants' data, 
    standardize selected features, and remove outlier observations.
    '''
    DF = pd.read_csv(config.raw_data_path)
    DF.rename(columns={"BASIC_INFO_ID": "ID", "BASIC_INFO_AGE": "Age"}, inplace=True) 
    elderly_idxs = DF["Age"] >= args.age_threshold
    elderly_DF = DF.loc[elderly_idxs, :].reset_index(drop=True)
    feats = elderly_DF[config.selected_features].dropna(axis=0, how='any')
    z_scores = (feats - feats.mean()) / feats.std(ddof=0)
    outlier_mask = (np.abs(z_scores) > 3).any(axis=1)
    final_DF = (
        elderly_DF.loc[:, ["ID", "Age"]]
        .merge(
            z_scores[~outlier_mask], # boolean indexing to remove outliers
            left_index=True, right_index=True
        )
    ).reset_index(drop=True)

    return final_DF

def build_design_matrix(DF, config, type=["train", "test"][0], mu=None, sigma=None):
    def _rename_features(f):
        f = f.replace("MOTOR_GOFITTS_", "GoFitts_")
        f = f.replace("MEMORY_EXCLUSION_", "Exclusion_")
        f = f.replace("MEMORY_OSPAN_", "OSpan_")
        f = f.replace("LANGUAGE_SPEECHCOMP_", "Speech_")
        f = f.replace("LANGUAGE_READING_", "Reading_")
        return f.replace("BEH_", "")
    
    X = DF.loc[:, config.selected_features] # linear terms
    X.rename(columns=_rename_features, inplace=True)

    if config.terms_included == "with-interaction":
        for i, xi in enumerate(X.columns):
            for j, xj in enumerate(X.columns):
                if j < i:
                    X[f"{i}_x_{j}"] = X[xi] * X[xj] # two-way interaction terms

    if type == "train":
        mu = X.mean().values
        sigma = X.std(ddof=0).values
        X = (X - mu) / sigma
    else:
        X = (X - mu) / sigma

    X.insert(0, "const", 1.0) # intercept

    return X, mu, sigma

def evaluate_model(y_true, y_pred):
    err = y_true - y_pred
    return {
        "MAE": np.mean(np.abs(err)), 
        "RMSE": np.sqrt(np.mean(err ** 2)), 
        "R2": 1 - ( np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2) )
    }

def run_loocv(DF, config):
    '''
    Run leave-one-out cross-validation.
    '''
    y_true, y_pred = [], []
    coefs_list = []

    for i in range(DF.shape[0]):
        DF_train = DF.drop(index=i).reset_index(drop=True)
        X_train, mu, sigma = build_design_matrix(DF_train, config, type="train")
        y_train = DF_train["Age"].values
        model = sm.OLS(y_train, X_train).fit()
        coefs_list.append(model.params)

        DF_test = DF.iloc[[i]]
        X_test, *_ = build_design_matrix(DF_test, config, type="test", mu=mu, sigma=sigma)
        y_pred.append(model.predict(X_test).values[0])
        y_true.append(DF_test["Age"])

    coefs = pd.concat(coefs_list, axis=1)
    coefs.index.name = "variable"
    coefs_summ = coefs.T.describe().loc[["min", "max", "mean", "std"]]
    coefs_summ.T.to_csv(config.loocv_coefs_path)

    res = evaluate_model(np.array(y_true), np.array(y_pred))
    with open(config.loocv_performance_path, "w", encoding="utf-8") as f:
        json.dump(convert_np_types(res), f, ensure_ascii=False)
        
def save_model_summ(model, config):
    # with open(config.model_summ_txt_path, "w", encoding="utf-8") as f:
    #     f.write(model.summary().as_text())

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_val": model.tvalues,
        "p_t": model.pvalues, 
        "p_sig": model.pvalues.map(lambda x: "*" * sum( x <= t for t in [0.05, 0.01, 0.001] )),
        "ci_lower": model.conf_int()[0],
        "ci_upper": model.conf_int()[1],
    })
    coef_df.index.name = "variable"
    coef_df.to_csv(config.model_coefs_path)

    model_info = {
        "n_obs": model.nobs,
        "df_model": model.df_model,
        "df_resid": model.df_resid,
        "R2": model.rsquared,
        "R2_adj": model.rsquared_adj,
        "AIC": model.aic,
        "BIC": model.bic,
        "LogLik": model.llf,
        "F_val": model.fvalue,
        "p_F": model.f_pvalue,
        "n_cond": model.condition_number,
        "cov_type": model.cov_type,
    }
    with open(config.model_summ_json_path, "w", encoding="utf-8") as f:
        json.dump(convert_np_types(model_info), f, ensure_ascii=False)

if __name__ == "__main__":
    args = define_arguments()
    config = Config(args)

    platform_features = platform_features()
    feature_domain_map = {0: "Motor", 1: "Memory", 2: "Language"}
    config.selected_domains = [ feature_domain_map[d] for d in args.domain ]
    config.selected_features = [ f for f in platform_features if any( d.upper() in f for d in config.selected_domains ) ]

    # if args.seed is None:
    #     config.seed = np.random.randint(0, 10000)
    # else:
    #     config.seed = args.seed
    # np.random.seed(config.seed)
    
    DF = prepare_data(config, args)
    DF.to_csv(config.data_path, index=False)
    config.N = DF.shape[0]
    save_config(config)

    run_loocv(DF, config)

    X, mu, sigma = build_design_matrix(DF, config, type="train")
    y = DF["Age"].values
    model = sm.OLS(y, X).fit()
    save_model_summ(model, config)



    

