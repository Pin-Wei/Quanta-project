#!/usr/bin/python

import os
import argparse
import logging
import copy
import shutil
import json
import pickle
from datetime import datetime
from itertools import product
# from collections import defaultdict

import numpy as np
import pandas as pd
import shap # SHapley Additive exPlanations
import optuna
import optunahub
# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform

from imblearn.over_sampling import SMOTENC 
from imblearn.under_sampling import RandomUnderSampler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, ElasticNetCV, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

## Argument parser: ===================================================================

parser = argparse.ArgumentParser(description="")
## How to split data into groups:
parser.add_argument("-age", "--age_method", type=int, default=2, 
                    help="The method to define age groups (0: 'no_cut', 1: 'cut_at_40', 2: 'cut_44-45', 3: 'wais_8_seg').")
parser.add_argument("-sex", "--by_gender", type=int, default=1, 
                    help="Whether to separate the data by gender (0: False, 1: True).")
parser.add_argument("-upd", "--use_prepared_data", type=str, default=None, 
                    help="File path of the data to be used (.csv).")
## Balancing data such that all groups have the same number of participants:
parser.add_argument("-u", "--smotenc", action="store_true", default=False, 
                    help="Up-sample the data using SMOTENC.")
parser.add_argument("-d", "--downsample", action="store_true", default=False, 
                    help="Down-sample the data without replacement.")
parser.add_argument("-b", "--bootstrap", action="store_true", default=False, 
                    help="Down-sample the data with replacement (i.e., bootstrapping).")
parser.add_argument("-bg", "--balancing_groups", type=int, default=0, 
                    help="The groups to be balanced.")
parser.add_argument("-n", "--sample_size", type=int, default=None, 
                    help="The number of participants to up- or down-sample to. (if None, use the default number of participants per group set in constants.n_per_balanced_g).")
## Split data into training and testing sets:
parser.add_argument("-tsr", "--testset_ratio", type=float, default=0.3, 
                    help="The ratio of the testing set.")
## Feature selection:
parser.add_argument("-iam", "--include_all_mappings", action="store_true", default=False, 
                    help="Include 'All' domain-approach mappings for feature selection.")
parser.add_argument("-oam", "--only_all_mapping", action="store_true", default=False, 
                    help="Include only 'All' domain-approach mappings for feature selection.")
parser.add_argument("-psf", "--preselected_feature_folder", type=str, default=None, 
                    help="The folder where the result files (.json) containing the selected features are stored.")
parser.add_argument("-fsm", "--feature_selection_method", type=int, default=0, 
                    help="The method to select features.")
parser.add_argument("-fst", "--fs_thresh_method", type=int, default=1, 
                    help="The method to determine the threshold for feature selection (0: 'threshold', 1: 'explained_ratio').")
parser.add_argument("-mfn", "--max_feature_num", type=int, default=None, 
                    help="The maximum number of features to be selected.")
# parser.add_argument("--no_pca", action="store_true", default=False, 
#                     help="Do not use PCA for feature selection.")
# parser.add_argument("-epr", "--explained_ratio", type=float, default=0.9, 
#                     help="The variance to be explained by the selected features.")
## Model training:
parser.add_argument("-pmf", "--pretrained_model_folder", type=str, default=None, 
                    help="The folder where the pre-trained model files (.pkl) are stored.")
parser.add_argument("-m", "--training_model", type=int, default=None, 
                    help="The type of the model to be used for training (0: 'ElasticNet', 1: 'RF', 2: 'CART', 3: 'LGBM', 4: 'XGBM').")
parser.add_argument("-i", "--ignore", type=int, default=0, 
                    help="Ignore the first N iterations (in case the script was interrupted by an accident and you don't want to start from the beginning).")
parser.add_argument("-s", "--seed", type=int, default=None, 
                    help="The value used to initialize all random number generator.")
## Age correction:
parser.add_argument("-acm", "--age_correction_method", type=int, default=0, 
                    help="The method to correct age (0: 'Zhang et al. (2023)', 1: 'Beheshti et al. (2019)').")
args = parser.parse_args()

## Classes: ===========================================================================

class Config:
    def __init__(self):
        self.source_path = os.path.dirname(os.path.abspath(__file__))
        self.data_file_path = os.path.join(self.source_path, "rawdata", "DATA_ses-01_2024-12-09.csv")
        self.inclusion_file_path = os.path.join(self.source_path, "rawdata", "InclusionList_ses-01.csv")
        self.balancing_groups = ["wais_8_seg", "cut_44-45"][args.balancing_groups]
        self.age_method = ["no_cut", "cut_at_40", "cut_44-45", "wais_8_seg"][args.age_method]
        self.by_gender = [False, True][args.by_gender]
        self.testset_ratio = args.testset_ratio
        self.feature_selection_method = [None, "LassoCV", "ElasticNetCV", "RF-Permute", "LGBM-SHAP"][args.feature_selection_method]
        self.fs_thresh_method = ["ftxed_threshold", "explained_ratio"][args.fs_thresh_method]
        self.age_correction_method = ["Zhang et al. (2023)", "Beheshti et al. (2019)"][args.age_correction_method]
        self.age_correction_groups = ["wais_8_seg", "every_5_yrs"][0]

        folder_prefix = datetime.today().strftime('%Y-%m-%d')
        if args.use_prepared_data:
            syn_method = os.path.basename(os.path.dirname(args.use_prepared_data)).split("_")[0]
            folder_prefix += f"_{syn_method}"
        elif args.smotenc:
            folder_prefix += "_smotenc" # up-sampled
        elif args.downsample:
            folder_prefix += "_down-sampled"
        elif args.bootstrap:
            folder_prefix += "_bootstrapped"
        else:
            folder_prefix += "_original"

        # if args.seed is not None:
        #     folder_prefix += f"_seed={args.seed}"

        if args.age_method == 0:
            folder_prefix += "_age-0"

        if args.by_gender == 0:
            folder_prefix += "_sex-0"

        if args.pretrained_model_folder is not None:
            if len(args.pretrained_model_folder) > 30: # avoid too long path
                self.out_folder = os.path.join(self.source_path, "outputs", f"{folder_prefix}_pre-trained")
            else:
                self.out_folder = os.path.join(self.source_path, "outputs", f"{folder_prefix} ({args.pretrained_model_folder})")
        else:
            self.out_folder = os.path.join(self.source_path, "outputs", folder_prefix)

        while os.path.exists(self.out_folder): # make sure the output folder does not exist:
            self.out_folder = self.out_folder + "+"

        self.description_outpath = os.path.join(self.out_folder, "description.json")
        self.prepared_data_outpath = os.path.join(self.out_folder, "prepared_data.csv")
        self.logging_outpath = os.path.join(self.out_folder, "log.txt")
        self.failure_record_outpath = os.path.join(self.out_folder, "failure_record.txt")
        self.scaler_outpath_format = os.path.join(self.out_folder, "scaler_{}.pkl")
        self.model_outpath_format = os.path.join(self.out_folder, "models_{}_{}_{}.pkl")
        self.features_outpath_format = os.path.join(self.out_folder, "features_{}_{}_{}.csv")
        self.training_results_outpath_format = os.path.join(self.out_folder, "training_results_{}_{}.json")
        self.results_outpath_format = os.path.join(self.out_folder, "results_{}_{}.json")

class Constants:
    def __init__(self):
        ## The age groups defined by different methods:
        self.age_groups = { 
            "no_cut": {
                "all": (0, np.inf)
            }, 
            "cut_at_40": {
                "le-40" : ( 0, 40),    # less than or equal to
                "ge-41" : (41, np.inf) # greater than or equal to
            }, 
            "cut_44-45": {
                "le-44" : ( 0, 44),
                "ge-45" : (45, np.inf) 
            }, 
            "wais_8_seg": {
                "le-24": ( 0, 24), 
                "25-29": (25, 29), 
                "30-34": (30, 34),
                "35-44": (35, 44), 
                "45-54": (45, 54), 
                "55-64": (55, 64), 
                "65-69": (65, 69), 
                "ge-70": (70, np.inf)
            }, 
            "every_5_yrs": {
                "le-24": ( 0, 24), 
                "25-29": (25, 29), 
                "30-34": (30, 34), 
                "35-39": (35, 39), 
                "40-44": (40, 44), 
                "45-49": (45, 49), 
                "50-54": (50, 54), 
                "55-59": (55, 59), 
                "60-64": (60, 64), 
                "65-69": (65, 69), 
                "70-74": (70, 74), 
                "ge-75": (75, np.inf)
            }
        }
        ## The correspondence between domains and approaches:
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
            }
        }
        ## The names of models to evaluate:
        self.model_names = [ 
            "ElasticNet", 
            "RF",   # RandomForestRegressor
            "CART", # DecisionTreeRegressor
            "LGBM", # lgb.LGBMRegressor
            "XGBM"  # xgb.XGBRegressor
        ]
        ## The number of participants in each balanced group:
        self.n_per_balanced_g = {
            "wais_8_seg": {
                "SMOTENC": 60, 
                "downsample": 15, 
                "bootstrap": 15
            }, 
            "cut_44-45": {
                "SMOTENC": 240, 
                "downsample": 78, 
                "bootstrap": 78
            }
        }
        ## Number of parallel threads to use:
        self.n_jobs = 16

class FeatureSelector:
    def __init__(self, method, thresh_method, threshold, explained_ratio, max_feature_num, seed, n_jobs=16):
        self.method = method
        self.seed = seed
        self.thresh_method = thresh_method
        self.threshold = threshold
        self.explained_ratio = explained_ratio
        self.max_feature_num = max_feature_num
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y):
        '''
        Select features based on their importance weights.
        - see: https://scikit-learn.org/stable/modules/feature_selection.html
        - also: https://hyades910739.medium.com/%E6%B7%BA%E8%AB%87-tree-model-%E7%9A%84-feature-importance-3de73420e3f2

        Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
         + handling multicollinearity: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
         - may not be suitable

        SHAP (SHapley Additive exPlanations): https://shap.readthedocs.io/en/latest/
        - see 1: https://christophm.github.io/interpretable-ml-book/shapley.html
        - see 2: https://ithelp.ithome.com.tw/articles/10329606
        - see 3: https://medium.com/analytics-vidhya/shap-part-1-an-introduction-to-shap-58aa087a460c
        - see 4: https://medium.com/@msvs.akhilsharma/unlocking-the-power-of-shap-analysis-a-comprehensive-guide-to-feature-selection-f05d33698f77
        '''
        if self.method == "LassoCV":
            importances = LassoCV(
                cv=5, random_state=self.seed
            ).fit(X, y).coef_

        elif self.method == "ElasticNetCV":
            importances = ElasticNetCV(
                l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 
                cv=5, random_state=self.seed, n_jobs=self.n_jobs
            ).fit(X, y).coef_

        elif self.method == "RF-Permute": # permutation importance 
            # dist_matrix = 1 - X.corr(method="spearman").abs()
            # dist_linkage = hierarchy.ward( # compute Ward’s linkage on a condensed distance matrix.
            #     squareform(dist_matrix)
            # ) 
            # cluster_ids = hierarchy.fcluster( # form flat clusters from the hierarchical clustering defined by the given linkage matrix
            #     Z=dist_linkage, t=2, criterion="distance" # t: distance threshold, manually selected
            # )
            # cid_to_fids = defaultdict(list) # cluster id to feature ids
            # for idx, cluster_id in enumerate(cluster_ids):
            #     cid_to_fids[cluster_id].append(idx)
            # first_features = [ v[0] for v in cid_to_fids.values() ] # select the first feature in each cluster
            X_train, X_test, y_train, y_test = train_test_split(
                # X.iloc[:, first_features], y, test_size=.3, random_state=self.seed
                X, y, test_size=.2, random_state=self.seed
            )
            rf_trained = RandomForestRegressor(
                random_state=self.seed, n_jobs=self.n_jobs
            ).fit(X_train, y_train)
            importances = permutation_importance(
                estimator=rf_trained, X=X_test, y=y_test, 
                n_repeats=10, random_state=self.seed, n_jobs=self.n_jobs
            ).importances_mean

        elif self.method == "LightGBM": # impurity-based feature importance
            # importances = LGBMRegressor(
            #     importance_type=["split", "gain"][1], random_state=self.seed
            # ).fit(X, y).feature_importances_
            raise NotImplementedError("LightGBM impurity-based feature importance should not be used.")

        elif self.method == "LGBM-SHAP": 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.2, random_state=self.seed
            )
            lgbm_trained = LGBMRegressor(
                min_child_samples=5, random_state=self.seed, n_jobs=self.n_jobs
            ).fit(X_train, y_train)
            shap_values = shap.GPUTreeExplainer(lgbm_trained).shap_values(X_test)
            importances = np.abs(shap_values).mean(axis=0)

        feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        
        if self.thresh_method == "explained_ratio":
            normed_feature_imp = feature_importances / feature_importances.abs().sum()
            cumulative_importances = normed_feature_imp.abs().cumsum()
            num_features = np.argmax(cumulative_importances > self.explained_ratio) + 1
            selected_feature_imp = feature_importances.head(num_features)

        elif self.thresh_method == "threshold":
            selected_feature_imp = feature_importances[feature_importances.abs() > self.threshold]

        if self.max_feature_num is not None:
            selected_feature_imp = selected_feature_imp.head(self.max_feature_num)
           
        self.feature_importances = selected_feature_imp
        self.selected_features = list(selected_feature_imp.index)

        return self
    
    def transform(self, X):
        return X[self.selected_features]

## Functions: =========================================================================

def load_and_merge_datasets(data_file_path, inclusion_file_path):
    '''
    Read the data and inclusion table, and merge them.
    '''
    ## Load the main dataset:
    DF = pd.read_csv(data_file_path)
    
    ## Load the file marking whether a data has been collected from individual participants:
    inclusion_df = pd.read_csv(inclusion_file_path)

    ## Only include participants with MRI data:
    inclusion_df = inclusion_df.query("MRI == 1")

    ## Ensure consistent ID column names:
    if "BASIC_INFO_ID" in DF.columns:
        DF = DF.rename(columns={"BASIC_INFO_ID": "ID"})

    ## Merge the two dataframes to apply inclusion criteria:
    DF = pd.merge(DF, inclusion_df[["ID"]], on="ID", how='inner')

    ## Transform column data type:
    DF["BASIC_INFO_SEX"] = DF["BASIC_INFO_SEX"].astype('int')
    DF["BASIC_INFO_AGE"] = DF["BASIC_INFO_AGE"].astype('int')

    return DF

def make_balanced_dataset(DF, balancing_method, age_bin_dict, n_per_balanced_g, seed):
    '''
    Make balanced datasets using specified balancing methods, including:
    
    OverSampling using 'SMOTENC' (Synthetic Minority Oversampling Technique)
    - see: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html#imblearn.over_sampling.SMOTENC.fit_resample
    
    UnderSampling using 'RandomUnderSampler' with (bootstrap) or without replacement (downsample)
    - see: https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html#imblearn.under_sampling.RandomUnderSampler
    '''
    ## Assign "AGE-GROUP_SEX" labels:
    DF["AGE-GROUP"] = pd.cut(
        x=DF["BASIC_INFO_AGE"], 
        bins=[ 0 ] + [ x for _, x in list(age_bin_dict.values()) ], 
        labels=list(age_bin_dict.keys())
    )
    DF["SEX"] = DF["BASIC_INFO_SEX"].map({1: "M", 2: "F"})
    DF["AGE-GROUP_SEX"] = DF.loc[:, ["AGE-GROUP", "SEX"]].agg("_".join, axis=1)

    ## Drop redundant columns:
    DF.drop(columns=["ID", "AGE-GROUP", "SEX"], inplace=True)

    ## Drop and "BASIC_INFO_" columns except "BASIC_INFO_AGE":
    DF.drop(columns=[ col for col in DF.columns if (( col.startswith("BASIC_") ) and ( col not in ["BASIC_INFO_AGE", "BASIC_INFO_SEX"] ))], inplace=True)
    
    ## Fill missing values:
    target_col = "AGE-GROUP_SEX"
    target_classes = list(DF[target_col].unique())
    DF_imputed_list = []
    for t in target_classes:
        sub_DF = DF[DF[target_col] == t]
        sub_X = sub_DF.drop(columns=[target_col])
        imputer = SimpleImputer(strategy="median")
        sub_DF_imputed = pd.DataFrame(imputer.fit_transform(sub_X), columns=sub_X.columns)
        sub_DF_imputed[target_col] = sub_DF[target_col].reset_index(drop=True)
        DF_imputed_list.append(sub_DF_imputed)
    DF_imputed = pd.concat(DF_imputed_list)
    DF_imputed.reset_index(drop=True, inplace=True)

    ## Make balanced datasets:   
    if balancing_method == "CTGAN":
        raise NotImplementedError("CTGAN is not implemented yet.")
    elif balancing_method == "SMOTENC":
        sampler = SMOTENC(
            categorical_features=["BASIC_INFO_AGE", "BASIC_INFO_SEX"], 
            sampling_strategy={ t: n_per_balanced_g for t in target_classes }, 
            random_state=seed
        )
    elif balancing_method == "downsample":
        sampler = RandomUnderSampler(
            sampling_strategy={ t: n_per_balanced_g for t in target_classes }, 
            random_state=seed, 
            replacement=False
        ) 
    elif balancing_method == "bootstrap":
        sampler = RandomUnderSampler(
            sampling_strategy={ t: n_per_balanced_g for t in target_classes }, 
            random_state=seed, 
            replacement=True
        ) 
    X_resampled, y_resampled = sampler.fit_resample(
        X=DF_imputed.drop(columns=[target_col]), 
        y=DF_imputed[target_col]
    )

    ## Merge back with target variable and create ID column:
    DF_balanced = pd.merge(
        pd.DataFrame({target_col: y_resampled}), X_resampled, 
        left_index=True, right_index=True
    )
    DF_balanced.insert(0, "ID", [ f"sub-{x:04d}" for x in DF_balanced.index ])

    return target_col, DF_balanced

def mark_synthetic_data(DF, DF_upsampled):
    '''
    Add a new column to mark whether the data is real or synthetic.
    '''
    ## Find the common NA-free columns between the two dataframes:
    DF_nona = DF.dropna(axis=1)
    nona_cols = list(DF_nona.columns)
    common_nona_cols = [ x for x in nona_cols if x in DF_upsampled.columns ]
    common_nona_cols.remove("ID")

    ## Use inner join on common_nona_cols to find real data:
    DF_real = ( 
        DF_upsampled
        .loc[:, common_nona_cols]
        .reset_index()
        .merge(DF.loc[:, common_nona_cols], how='inner', on=common_nona_cols)
        .set_index('index')
    )

    ## Add a new column to mark whether the data is real or synthetic:
    DF_marked = DF_upsampled.copy(deep=True) # avoid modifying the original dataframe
    DF_marked.insert(1, "R_S", "Synthetic")
    DF_marked.loc[DF_real.index, "R_S"] = "Real"

    return DF_marked

def preprocess_grouped_dataset(X, y, ids, testset_ratio, trained_scaler, seed):
    '''
    Fill missing values, split into training and testing sets, and feature scale the grouped dataset.
    
    Inputs:
    - X (pd.DataFrame): Feature matrix
    - y (pd.Series): Target variable
    - ids (pd.Series): IDs of participants
    - testset_ratio (float): The ratio of testing set to the whole dataset
    - seed (int): Random seed
    
    Return:
    - (dict): A dictionary of train-test splited data, storing 
              the standardized feature values, age, and ID numbers of participants.
    '''
    ## Fill missing values:
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    ## Split into training and testing sets, and then apply feature scaling:
    if trained_scaler is None:
        scaler = MinMaxScaler()

    if testset_ratio != 0:
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X_imputed, y, ids, test_size=testset_ratio, random_state=seed)
        if trained_scaler is None:
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        else:
            X_train_scaled = pd.DataFrame(trained_scaler.transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(trained_scaler.transform(X_test), columns=X_test.columns)
    else:
        X_train, y_train, id_train = X_imputed, y, ids
        if trained_scaler is None:
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        else:
            X_train_scaled = pd.DataFrame(trained_scaler.transform(X_train), columns=X_train.columns)
        X_test_scaled, y_test, id_test = pd.DataFrame(), pd.Series(), pd.Series()

    return {
        "X_train": X_train_scaled, 
        "X_test": X_test_scaled, 
        "y_train": y_train, 
        "y_test": y_test, 
        "id_train": id_train.reset_index(drop=True), 
        "id_test": id_test.reset_index(drop=True), 
        "scaler": scaler if trained_scaler is None else trained_scaler
    }

def build_pipline(params, model_name, seed, n_jobs=16):
    '''
    Return:
    - (sklearn.pipeline.Pipeline): A pipeline of feature selection and regression model.
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    '''
    ## Feature selection:
    selector = FeatureSelector(
        method=params["method"], 
        thresh_method=params["thresh_method"], 
        threshold=params["threshold"], 
        explained_ratio=params["explained_ratio"],
        max_feature_num=params["max_feature_num"], 
        seed=seed, 
        n_jobs=n_jobs
    )
    
    ## Regression model:
    if model_name == "ElasticNet":
        model = ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            random_state=seed
        )

    elif model_name == "CART":
        model = DecisionTreeRegressor(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=seed
        )

    elif model_name == "RF":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=seed,
            n_jobs=n_jobs
        )

    elif model_name == "XGBM":
        model = XGBRegressor(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=seed,
            n_jobs=n_jobs
        )

    elif model_name == "LGBM":
        model = LGBMRegressor(
            metric='mae',
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            feature_fraction=params["feature_fraction"],
            bagging_fraction=params["bagging_fraction"],
            bagging_freq=params["bagging_freq"],
            min_child_samples=params["min_child_samples"],
            random_state=seed,
            n_jobs=n_jobs
        )

    return Pipeline([
        ("feature_selector", selector), 
        ("regressor", model)
    ])

def optimize_objective(trial, X, y, fs_method, thresh_method, max_feature_num, model_name, seed, n_jobs=16): 
    '''
    Return:
    - (float): Average mean absolute error (MAE) score across cross validation.
    '''
    ## Set up feature selection parameters:
    if fs_method is None:
        params = {
            "method": trial.suggest_categorical("fs_method", ["LassoCV", "ElasticNetCV", "RF-Permute", "LGBM-SHAP"]), 
            "thresh_method": thresh_method, 
            "max_feature_num": max_feature_num
        }
    else:
        params = {
            "method": fs_method, 
            "thresh_method": thresh_method, 
            "max_feature_num": max_feature_num
        }

    if params["thresh_method"] == "ftxed_threshold":
        params.update({
            "threshold": trial.suggest_float("threshold", 1e-5, 0.001), 
            "explained_ratio": None
        })

    elif params["thresh_method"] == "explained_ratio":
        params.update({
            "explained_ratio": trial.suggest_float('explained_ratio', 0.9, 1), 
            "threshold": None
        })

    ## Record feature selection parameters:
    for k, v in params.items():
        if v is not None:
            trial.set_user_attr(f"fs_{k}", v)
    
    ## Set up model parameters:
    if model_name == "ElasticNet":
        params.update({
            "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
        })

    elif model_name == "CART":
        params.update({
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
        })

    elif model_name == "RF":
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 32)
        })

    elif model_name == "XGBM":
        params.update({
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0)
        })

    elif model_name == "LGBM":
        params.update({
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
        })

    pipline = build_pipline(
        params, model_name, seed, n_jobs
    )
    neg_mae_scores = cross_val_score(
        estimator=pipline, X=X, y=y, 
        cv=KFold(n_splits=5, shuffle=True, random_state=seed), 
        scoring="neg_mean_absolute_error", 
        n_jobs=n_jobs, 
        verbose=1
    )

    return -1 * np.mean(neg_mae_scores)

def train_and_evaluate(X, y, fs_method, thresh_method, max_feature_num, model_names, seed, n_jobs=16):
    '''
    Inputs:
    - X (pd.DataFrame)  : Feature matrix.
    - y (pd.Series)     : Target variable (i.e., age).
    - model_names (list): List of keys that specify the models to evaluate.

    Return:
    - (dict): The evaluation results for each model, across cross validation.
    '''
    results = {}

    for model_name in model_names:
        logging.info(f"Optimizing hyperparameters for {model_name} ...")

        ## Initialize the model with the best hyperparameters:
        study = optuna.create_study(
            direction='minimize', 
            # sampler=optuna.samplers.TPESampler(seed=seed), 
            sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler() 
                ## automatically selects an algorithm internally, see: https://medium.com/optuna/autosampler-automatic-selection-of-optimization-algorithms-in-optuna-1443875fd8f9
        )
        study.optimize(
            lambda trial: optimize_objective(
                trial, X, y, fs_method, thresh_method, max_feature_num, model_name, seed, n_jobs
            ),
            n_trials=50, show_progress_bar=True
        )
        logging.info("Parameter optimization is completed :-)")

        ## Refit the model with the best hyperparameters found:
        best_pipeline = build_pipline(
            study.best_params, model_name, seed, n_jobs
        )
        mae_scores = -1 * cross_val_score(
            estimator=best_pipeline, X=X, y=y, 
            cv=KFold(n_splits=5, shuffle=True, random_state=seed), 
            scoring="neg_mean_absolute_error", 
            n_jobs=n_jobs, 
            verbose=1
        )

        ## Storing results:
        results[model_name] = {
            "trained_model": best_pipeline.named_steps["regressor"],  
            "selected_features": best_pipeline.named_steps["feature_selector"].selected_features, 
            "feature_importances": best_pipeline.named_steps["feature_selector"].feature_importances, 
            "cv_scores": mae_scores, 
            "mae_mean": np.mean(mae_scores), 
            "mae_std": np.std(mae_scores)
        }
        results[model_name] = results[model_name].update(study.best_params)

    return results

def generate_correction_ref(age, pad, age_groups, age_breaks): 
    '''
    Inputs:
    - age (pd.Series): The actual age of the participants.
    - pad (np.Array): Predicted age difference.
    - age_groups (list): The labels of age groups.
    - age_breaks (sequence of scalars): The bin edges of age groups.

    Return:
    - (pd.DataFrame): A reference table for age correction, including 
        the mean and standard deviation of PAD for each age group.
    '''
    ## Create a DataFrame containing the actual age and PAD values:
    DF = pd.DataFrame({"Age": age, "PAD": pad})

    ## Assign age labels based on the given age bin edges:
    DF["Group"] = pd.cut(DF["Age"], bins=age_breaks, labels=age_groups)

    ## Calculate the mean and standard deviation of PAD for each age group:
    correction_ref = DF.groupby("Group", observed=True)["PAD"].agg(['mean', 'std']).reset_index()
    
    return correction_ref.rename(columns={"mean": "PAD_mean", "std": "PAD_std"})  

def apply_age_correction(predictions, true_ages, correction_ref, age_groups, age_breaks):
    '''
    Inputs:
    - predictions (pd.Series): Predicted age.
    - true_ages (pd.Series): Actual age.
    - correction_ref (pd.DataFrame): Reference table for age correction.

    Return:
    - (np.Array): Corrected age predictions.
    '''
    corrected_predictions = []

    for pred, true_age in zip(predictions, true_ages):

        ## Calculate the predicted age difference (PAD) for the current sample:
        pad = pred - true_age

        ## Determine the age group of the current sample:
        age_label = pd.cut([true_age], bins=age_breaks, labels=age_groups)[0]

        ## Get the mean and standard deviation of PAD for the age group:
        if age_label in list(correction_ref["Group"]):
            pad_mean = correction_ref.query("Group == @age_label")["PAD_mean"].values[0]
            pad_std = correction_ref.query("Group == @age_label")["PAD_std"].values[0]
        else:
            ## If the age group is not in the reference table, use the mean and standard deviation of all samples:
            pad_mean = correction_ref["PAD_mean"].mean()
            pad_std = correction_ref["PAD_std"].mean()

        if pad_std == 0: # Handle the case where the std of the PAD is zero
            padac = pad - pad_mean
        else:
            padac = (pad - pad_mean) / pad_std # The age-corrected PAD

        ## Store the corrected age prediction into list:
        corrected_predictions.append(pred - padac)

    return np.array(corrected_predictions)

def model_age_related_bias(real_ages, offsets):
    reg  = LinearRegression().fit(
        X=np.array(real_ages).reshape(-1, 1), y=np.array(offsets)
    )
    return {"intercept": reg.intercept_, "slope": reg.coef_}

def calc_bias_free_offsets(reg, predicted_ages):
    return predicted_ages * reg["slope"] + reg["intercept"]

def convert_np_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert np.ndarray to list
    elif isinstance(obj, np.generic): 
        return obj.item()   # Convert np.generic to scalar
    elif isinstance(obj, pd.Series):
        return obj.tolist() # Convert pd.Series to list
    elif isinstance(obj, list):
        return [ convert_np_types(i) for i in obj ] 
    elif isinstance(obj, dict):
        return { k: convert_np_types(v) for k, v in obj.items() } 
    else:
        return obj

## Main: ==============================================================================

def main():
    ## Setup config and constant objects:
    config = Config()
    constant = Constants()

    os.makedirs(config.out_folder) # should not exist

    ## Setup logging:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename=config.logging_outpath
    ) # https://zx7978123.medium.com/python-logging-%E6%97%A5%E8%AA%8C%E7%AE%A1%E7%90%86%E6%95%99%E5%AD%B8-60be0a1a6005

    ## Define the random seed:
    if args.seed is None:
        seed = 9865 # np.random.randint(0, 10000)
    else:
        seed = args.seed

    ## Define the sampling method and number of participants per balanced group:
    if args.use_prepared_data: # should be balanced
        folder, _ = os.path.split(args.use_prepared_data)
        balancing_method, balancing_groups, n_per_balanced_g = "--", "--", "--" # default
        try:
            with open(os.path.join(folder, "description.json"), 'r', errors='ignore') as f:
                desc_json = json.load(f)
            balancing_method = desc_json["DataBalancingMethod"]
            balancing_groups = desc_json["BalancingGroups"]
            n_per_balanced_g = desc_json["NumPerBalancedGroup"]
        except:
            pass
    else:
        if args.smotenc:
            balancing_method = "SMOTENC"
        elif args.downsample:
            balancing_method = "downsample"
        elif args.bootstrap:
            balancing_method = "bootstrap"
        else: # no balancing
            balancing_method = None
            balancing_groups = None
            n_per_balanced_g = None

        if balancing_method is not None: # going to be balanced
            balancing_groups = config.balancing_groups
            if args.sample_size is None:
                n_per_balanced_g = constant.n_per_balanced_g[balancing_groups][balancing_method]
            else:
                n_per_balanced_g = args.sample_size

    ## Define the labels and boundaries of age groups:
    age_bin_labels = list(constant.age_groups[config.age_method].keys())
    age_boundaries = list(constant.age_groups[config.age_method].values()) 

    ## Define the labels and boundaries for age correction:
    pad_age_groups = list(constant.age_groups[config.age_correction_groups].keys())
    pad_age_breaks = [ 0 ] + [ x for _, x in list(constant.age_groups[config.age_correction_groups].values()) ] 

    ## Include (or only include) the 'ALL' domain-approach mapping, if specified:
    if args.include_all_mappings:
        logging.info("Include the 'ALL' domain-approach mapping.")
        constant.domain_approach_mapping["ALL"] = {
            "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
            "approaches": ["MRI", "BEH", "EEG"]
        }
    elif args.only_all_mapping:
        logging.info("Only include the 'ALL' domain-approach mapping.")
        constant.domain_approach_mapping = {
            "ALL": {
                "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
                "approaches": ["MRI", "BEH", "EEG"]
            }
        }

    ## Define the type of the model to be used for training:
    if args.training_model is not None:
        included_models = [constant.model_names[args.training_model]]
        print_included_models = included_models
    elif args.pretrained_model_folder is not None:
        print_included_models = "Depend on the previous results"
    else:
        included_models = constant.model_names
        print_included_models = included_models

    ## Save the description of the current execution as a JSON file:
    desc = {
        "Seed": seed, 
        "UsePreparedData": args.use_prepared_data, 
        "RawDataVersion": config.data_file_path if args.use_prepared_data is None else "--", 
        "InclusionFileVersion": config.inclusion_file_path if args.use_prepared_data is None else "--", 
        "DataBalancingMethod": balancing_method, 
        "BalancingGroups": balancing_groups,
        "NumPerBalancedGroup": n_per_balanced_g, 
        "SexSeparated": config.by_gender, 
        "AgeGroups": age_bin_labels, 
        "TestsetRatio": config.testset_ratio, 
        "UsePretrainedModels": args.pretrained_model_folder, 
        "UsePreviouslySelectedFeatures": args.preselected_feature_folder, 
        "FeatureOrientations": list(constant.domain_approach_mapping.keys()), 
        "FeatureSelectionMethod": config.feature_selection_method, 
        "FSThresholdMethod": config.fs_thresh_method,
        "MaxFeatureNum": args.max_feature_num, 
        "IncludedOptimizationModels": print_included_models, 
        "SkippedIterationNum": args.ignore,  
        "AgeCorrectionMethod": config.age_correction_method, 
        "AgeCorrectionGroups": pad_age_groups
    }
    desc = convert_np_types(desc)
    with open(config.description_outpath, 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False)

    logging.info("The description of the current execution is saved :-)")

    ## Copy the current Python script to the output folder:
    shutil.copyfile(
        src=os.path.abspath(__file__), 
        dst=os.path.join(config.out_folder, os.path.basename(__file__))
    )
    logging.info("The current python script is copied to the output folder :-)")

    ## Record the failed processing to a text file:
    record_if_failed = [] 
    
    if args.use_prepared_data is not None: # Use the prepared dataset
        logging.info("Loading the prepared dataset ...")
        DF_prepared = pd.read_csv(args.use_prepared_data)
        if "R_S" in DF_prepared.columns:
            DF_prepared.to_csv(config.prepared_data_outpath.replace(".csv", " (marked).csv"), index=False)
            DF_prepared.drop(columns=["R_S"], inplace=True)
        else:
            DF_prepared.to_csv(config.prepared_data_outpath, index=False)
            
    else: # Load the raw dataset:
        logging.info("Loading the raw dataset and merging it with the inclusion table ...")
        DF = load_and_merge_datasets(
            data_file_path=config.data_file_path, 
            inclusion_file_path=config.inclusion_file_path
        )

        ## Make balanced datasets if specified:
        if balancing_method is not None:
            logging.info(f"Making balanced datasets using '{balancing_method}' method ...")
            target_col, DF_prepared = make_balanced_dataset(
                DF=copy.deepcopy(DF), 
                balancing_method=balancing_method, 
                age_bin_dict=constant.age_groups["wais_8_seg"], 
                n_per_balanced_g=n_per_balanced_g, 
                seed=seed
            )
            DF_prepared.drop(columns=[target_col], inplace=True)

            if args.smotenc:
                fn = config.prepared_data_outpath.replace(".csv", " (marked).csv")
                DF_marked = mark_synthetic_data(DF, DF_prepared)
                DF_marked.to_csv(fn, index=False)
        else:
            DF_prepared = DF

        ## Save it to file:
        DF_prepared.to_csv(config.prepared_data_outpath, index=False)

    ## Divide the dataset into groups and define their labels:
    if args.by_gender:
        logging.info("Separating data according to participants' age ranges and genders ...")
        sub_DF_list = [
            DF_prepared[(DF_prepared["BASIC_INFO_AGE"].between(lb, ub)) & (DF_prepared["BASIC_INFO_SEX"] == sex)] 
            for (lb, ub), sex in list(product(age_boundaries, [1, 2]))
        ]
        sub_DF_labels = [ f"{age_group}_{sex}" for age_group, sex in list(product(age_bin_labels, ["M", "F"])) ] 
    else:
        logging.info("Separating data according to participants' age ranges ...")
        sub_DF_list = [
            DF_prepared[DF_prepared["BASIC_INFO_AGE"].between(lb, ub)] 
            for lb, ub in age_boundaries
        ]
        sub_DF_labels = age_bin_labels

    ## Separately preprocess different data subsets: 
    preprocessed_data_dicts = {}

    for group_name, sub_DF in zip(sub_DF_labels, sub_DF_list):

        if sub_DF.empty:
            logging.warning(f"Oops! Data subset of the '{group_name}' group is empty :-S")
            preprocessed_data_dicts[group_name] = None

        else:
            logging.info(f"Preprocessing data subset of the {group_name} group ...")
            sub_DF = sub_DF.reset_index(drop=True)
            if args.pretrained_model_folder is not None:
                try:
                    with open(os.path.join("outputs", args.pretrained_model_folder, f"scaler_{group_name}.pkl"), 'rb') as f:
                        trained_scaler = pickle.load(f)
                    logging.info("Using pre-trained scaler.")
                except FileNotFoundError:
                    logging.warning("Pre-trained scaler not found, build a new MinMaxScaler.")
                    trained_scaler = None
            else:
                trained_scaler = None

            preprocessed_data_dicts[group_name] = preprocess_grouped_dataset(
                X=sub_DF.drop(columns=["ID", "BASIC_INFO_AGE"]), 
                y=sub_DF["BASIC_INFO_AGE"], 
                ids=sub_DF["ID"], 
                testset_ratio=config.testset_ratio, 
                trained_scaler=trained_scaler, 
                seed=seed
            ) # a dictionary of train-test splited data, storing 
              # the standardized feature values, age, and ID numbers of participants.

            ## Save the scaler to the output folder:
            with open(config.scaler_outpath_format.format(group_name), 'wb') as f:
                pickle.dump(preprocessed_data_dicts[group_name]["scaler"], f)

    logging.info("Preprocessing of all data subsets is completed.")
    logging.info("Starting loop through all groups and orientations ...")
    iter = 0 # iteration counter for skipping

    for group_name, data_dict in preprocessed_data_dicts.items():
        logging.info(f"Group: '{group_name}'")

        if data_dict is None: 
            logging.warning("Data subset of the current group is empty!!")
            logging.info("Unable to train models, skipping ...")
            record_if_failed.append(f"Entire {group_name}.")
            continue

        else: 
            for ori_name, ori_content in constant.domain_approach_mapping.items():
                logging.info(f"Feature orientation: {ori_name}")

                if iter < args.ignore: 
                    logging.info(f"Skipping {iter}-th iteration :-O")

                else: # do what supposed to be done
                    if args.pretrained_model_folder is not None:                       
                        logging.info("Using the pre-trained model :-P")

                        saved_json = os.path.join("outputs", args.pretrained_model_folder, f"results_{group_name}_{ori_name}.json")
                        with open(saved_json, 'r', encoding='utf-8') as f:
                            saved_results = json.load(f)

                        best_model_name = saved_results["Model"]
                        selected_features = saved_results["FeatureNames"]

                        saved_model = os.path.join("outputs", args.pretrained_model_folder, f"models_{group_name}_{ori_name}_{best_model_name}.pkl")
                        with open(saved_model, 'rb') as f:
                            trained_model = pickle.load(f)
                        
                        X_train_selected = data_dict["X_train"].loc[:, selected_features]

                    else: # do what supposed to be done
                        if args.preselected_feature_folder is not None: 
                            # logging.info("Using previously selected features :-P")

                            # saved_json = os.path.join("outputs", args.preselected_feature_folder, f"results_{group_name}_{ori_name}.json")
                            # with open(saved_json, 'r', errors='ignore') as f:
                            #     saved_results = json.load(f)

                            # selected_features = saved_results["FeatureNames"]
                            # best_model_name = saved_results["Model"]
                            # included_models = [ best_model_name ]
                            # logging.info(f"Since previously selected features are used, using the corresponding model type: {best_model_name}")
                            raise NotImplementedError("Not supported yet.")
                        
                        else: # do what supposed to be done
                            logging.info("Filtering features based on the domain and approach ...")
                            included_features = [ 
                                col for col in data_dict["X_train"].columns
                                if any( domain in col for domain in ori_content["domains"] )
                                and any( app in col for app in ori_content["approaches"] )
                                and "RESTING" not in col
                            ]
                            if ori_name == "FUNCTIONAL": # exclude "STRUCTURE" features
                                included_features = [ col for col in included_features if "STRUCTURE" not in col ]
                            
                            if len(included_features) == 0: 
                                logging.warning("No features are included for the current orientation, the definition may be wrong!!")
                                logging.info("Unable to train models, skipping ...")
                                record_if_failed.append(f"{ori_name} of {group_name}.")
                                continue

                            else: # good, go ahead
                                X_train_included = data_dict["X_train"].loc[:, included_features]
                                
                                logging.info("Training and evaluating models ...")
                                results = train_and_evaluate(
                                    X=X_train_included, 
                                    y=data_dict["y_train"], 
                                    fs_method=config.feature_selection_method, 
                                    thresh_method=config.fs_threshold_method, 
                                    max_feature_num=args.max_feature_num, 
                                    model_names=included_models, 
                                    seed=seed, 
                                    n_jobs=constant.n_jobs
                                ) 
                                best_model_name = min(
                                    results, key=lambda x: results[x]["mae_mean"]
                                )
                                trained_model = results[best_model_name]["trained_model"]
                                selected_features = results[best_model_name]["selected_features"]
                                
                                logging.info("Saving the best-performing model (.pkl) ...")
                                model_outpath = config.model_outpath_format.format(group_name, ori_name, best_model_name)
                                with open(model_outpath, 'wb') as f:
                                    pickle.dump(trained_model, f)

                                logging.info("Saving the best selected features and their importances ...")
                                features_outpath = config.features_outpath_format.format(group_name, ori_name, results[best_model_name]["fs_method"])
                                results[best_model_name]["feature_importances"].to_csv(
                                    features_outpath, header=False
                                )

                                logging.info("Saving other models to the 'other models' folder ...")
                                embedded_outpath = os.path.join(config.out_folder, "other models")
                                os.makedirs(embedded_outpath, exist_ok=True)
                                
                                for model_name, model_result in results.items():
                                    if model_name != best_model_name: 
                                        model_outpath = os.path.join(embedded_outpath, os.path.basename(config.model_outpath_format.format(group_name, ori_name, model_name)))
                                        with open(model_outpath, 'wb') as f:
                                            pickle.dump(model_result["trained_model"], f)
                                        # features_outpath = os.path.join(embedded_outpath, os.path.basename(config.features_outpath_format.format(group_name, ori_name, model_result["fs_method"]).replace(".csv", f"_{model_name}.csv")))
                                        # model_result["feature_importances"].to_csv(
                                        #     features_outpath, header=False
                                        # )

                                logging.info("Saving results for all models ...")
                                model_results = {
                                    model_name: {
                                        k: v for k, v in res.items() if k != "trained_model"
                                    } for model_name, res in results.items()
                                }
                                model_results = convert_np_types(model_results)
                                model_results_outpath = config.training_results_outpath_format.format(group_name, ori_name)
                                with open(model_results_outpath, 'w') as f:
                                    json.dump(model_results, f)
                            
                    logging.info("Applying the best model to the training set ...")
                    X_train_selected = data_dict["X_train"].loc[:, selected_features]
                    y_pred_train = trained_model.predict(X_train_selected)
                    pad_train = y_pred_train - data_dict["y_train"]

                    if config.age_correction_method == "Zhang et al. (2023)":
                        logging.info("Generating age-correction reference table ...")
                        correction_ref = generate_correction_ref(
                            age=data_dict["y_train"], 
                            pad=pad_train, 
                            age_groups=pad_age_groups, 
                            age_breaks=pad_age_breaks
                        )

                        logging.info("Applying age-correction to the training set ...")
                        corrected_y_pred_train = apply_age_correction(
                            predictions=y_pred_train, 
                            true_ages=data_dict["y_train"], 
                            correction_ref=correction_ref, 
                            age_groups=pad_age_groups, 
                            age_breaks=pad_age_breaks
                        )
                        corrected_y_pred_train = pd.Series(corrected_y_pred_train, index=data_dict["y_train"].index)
                        padac_train = corrected_y_pred_train - data_dict["y_train"]

                    elif config.age_correction_method == "Beheshti et al. (2019)":
                        logging.info("Applying age-correction to the training set using Beheshti et al.'s (2019) method ...")
                        correction_ref = model_age_related_bias(
                            data_dict["y_train"], pad_train
                        )
                        padac_train = calc_bias_free_offsets(
                            correction_ref, y_pred_train
                        )
                        corrected_y_pred_train = data_dict["y_train"] - padac_train

                    else:
                        logging.info("No age correction is applied ...")
                        corrected_y_pred_train = None
                        padac_train = None

                    if data_dict["X_test"].empty: 
                        save_results = {
                            "Model": best_model_name, 
                            "NumberOfSubjs": len(data_dict["id_train"]), 
                            "SubjID": list(data_dict["id_train"]), 
                            "Note": "Train and test sets are the same.", 
                            "Age": list(data_dict["y_train"]), 
                            "PredictedAge": list(y_pred_train), 
                            "PredictedAgeDifference": list(pad_train), 
                            "CorrectedPAD": list(padac_train), 
                            "CorrectedPredictedAge": list(corrected_y_pred_train), 
                            "AgeCorrectionTable": correction_ref.to_dict(orient='records') if type(correction_ref) is not dict else correction_ref, 
                            "NumberOfFeatures": len(selected_features), 
                            "FeatureNames": list(selected_features), 
                        }

                    else:
                        logging.info("Evaluating the best model on the testing set ...")
                        X_test_selected = data_dict["X_test"].loc[:, selected_features]                       
                        y_pred_test = trained_model.predict(X_test_selected)
                        y_pred_test = pd.Series(y_pred_test, index=data_dict["y_test"].index)
                        pad = y_pred_test - data_dict["y_test"]

                        if config.age_correction_method == "Zhang et al. (2023)":
                            logging.info("Applying age-correction to the testing set ...")
                            corrected_y_pred_test = apply_age_correction(
                                predictions=y_pred_test, 
                                true_ages=data_dict["y_test"], 
                                correction_ref=correction_ref, 
                                age_groups=pad_age_groups, 
                                age_breaks=pad_age_breaks
                            )
                            corrected_y_pred_test = pd.Series(corrected_y_pred_test, index=data_dict["y_test"].index)
                            padac = corrected_y_pred_test - data_dict["y_test"]

                        elif config.age_correction_method == "Beheshti et al. (2019)":
                            logging.info("Applying age-correction to the testing set using Beheshti et al.'s (2019) method ...")
                            padac = calc_bias_free_offsets(
                                correction_ref, y_pred_test
                            )
                            corrected_y_pred_test = data_dict["y_test"] - padac

                        else:
                            logging.info("Again, no age correction is applied ...")
                            corrected_y_pred_test = None
                            padac = None

                        save_results = {
                            "Model": best_model_name, 
                            "NumberOfTraining": len(data_dict["id_train"]), 
                            "TrainingSubjID": list(data_dict["id_train"]), 
                            "TrainingAge": list(data_dict["y_train"]), 
                            "TrainingPredAge": list(y_pred_train), 
                            "TrainingPAD": list(pad_train), 
                            "TrainingPADAC": list(padac_train), 
                            "TrainingCorPredAge": list(corrected_y_pred_train), 
                            "NumberOfTesting": len(data_dict["id_test"]), 
                            "TestingSubjID": list(data_dict["id_test"]), 
                            "Age": list(data_dict["y_test"]), 
                            "PredictedAge": list(y_pred_test), 
                            "PredictedAgeDifference": list(pad), 
                            "CorrectedPAD": list(padac), 
                            "CorrectedPredictedAge": list(corrected_y_pred_test), 
                            "AgeCorrectionTable": correction_ref.to_dict(orient='records') if type(correction_ref) is not dict else correction_ref, 
                            "NumberOfFeatures": len(selected_features), 
                            "FeatureNames": list(selected_features), 
                        }

                    save_results = convert_np_types(save_results)
                    results_outpath = config.results_outpath_format.format(group_name, ori_name)
                    with open(results_outpath, 'w', encoding='utf-8') as f:
                        json.dump(save_results, f, ensure_ascii=False)

                    logging.info("Modeling results are saved :-)")

                iter += 1

    with open(config.failure_record_outpath, 'w') as f:
        f.write("\n".join(record_if_failed))
    
    logging.info("The record of failed processing is saved.")    

if __name__ == "__main__":
    main()
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


