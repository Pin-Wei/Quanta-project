#!/usr/bin/python

# python age_pred_model.py [-age] [-sex] [-u] [-d] [-b] [-n] [-tsr] [-iam] [-oam] [-fsm] [-epr] [-pmf] [-i] [-s]

import os
import argparse
import logging
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import copy
from imblearn.over_sampling import SMOTENC 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import optuna
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import json

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
                    help="The number of participants to up- or down-sample to. (if None, use the default number of participants per group set in constants.N_per_group).")
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
parser.add_argument("-fsm", "--feature_selection_model", type=int, default=0, 
                    help="The model to use for feature selection (0: 'LassoCV', 1: 'RF', 2: 'XGBR').")
parser.add_argument("--no_pca", action="store_true", default=False, 
                    help="Do not use PCA for feature selection.")
parser.add_argument("-epr", "--explained_ratio", type=float, default=0.9, 
                    help="The variance to be explained by the selected features.")
## Model training:
parser.add_argument("-pmf", "--pretrained_model_folder", type=str, default=None, 
                    help="The folder where the pre-trained model files (.pkl) are stored.")
parser.add_argument("-m", "--training_model", type=int, default=None, 
                    help="The type of the model to be used for training (0: 'ElasticNet', 1: 'RF', 2: 'CART', 3: 'LGBM', 4: 'XGBM').")
parser.add_argument("-i", "--ignore", type=int, default=0, 
                    help="Ignore the first N iterations (in case you might be interrupted by an error and don't want to start from the beginning).")
parser.add_argument("-s", "--seed", type=int, default=None, 
                    help="The value used to initialize all random number generator.")
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
        self.feature_selection_model = ["LassoCV", "RF", "XGBR"][args.feature_selection_model]
        if not args.no_pca:
            self.explained_ratio = args.explained_ratio
        else:
            self.explained_ratio = None
        self.pad_method = ["wais_8_seg", "every_5_yrs"][0]

        if args.use_prepared_data:
            syn_method = os.path.basename(os.path.dirname(args.use_prepared_data)).split("_")[0]
            folder_prefix = f"{datetime.today().strftime('%Y-%m-%d')}_{syn_method}"
        elif args.smotenc:
            folder_prefix = f"{datetime.today().strftime('%Y-%m-%d')}_smotenc" # up-sampled
        elif args.downsample:
            folder_prefix = f"{datetime.today().strftime('%Y-%m-%d')}_down-sampled"
        elif args.bootstrap:
            folder_prefix = f"{datetime.today().strftime('%Y-%m-%d')}_bootstrapped"
        else:
            folder_prefix = f"{datetime.today().strftime('%Y-%m-%d')}_original"

        if args.pretrained_model_folder is not None:
            self.out_folder = os.path.join(self.source_path, "outputs", f"{folder_prefix} ({args.pretrained_model_folder})")
        elif args.seed is not None:
            self.out_folder = os.path.join(self.source_path, "outputs", f"{folder_prefix}_seed={args.seed}")
        else:
            self.out_folder = os.path.join(self.source_path, "outputs", f"{folder_prefix}_{datetime.today().strftime('%H.%M.%S')}")
        
        while os.path.exists(self.out_folder): # make sure the output folder does not exist:
            self.out_folder = self.out_folder + "+"

        self.description_outpath = os.path.join(self.out_folder, "description.json")
        self.prepared_data_outpath = os.path.join(self.out_folder, "prepared_data.csv")
        self.logging_outpath = os.path.join(self.out_folder, "log.txt")
        self.failure_record_outpath = os.path.join(self.out_folder, "failure_record.txt")
        self.model_outpath_format = os.path.join(self.out_folder, "models_{}_{}_{}.pkl")
        self.model_maes_outpath_format = os.path.join(self.out_folder, "model_maes_{}_{}.json")
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
                # "ge-65": (65, np.inf)
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
        self.N_per_group = {
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
        ## The number of features to select (if not using PCA):
        self.feature_num = 50

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

def make_balanced_dataset(DF, balancing_method, age_bin_dict, N_per_group, seed):
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
            sampling_strategy={ t: N_per_group for t in target_classes }, 
            random_state=seed
        )
    elif balancing_method == "downsample":
        sampler = RandomUnderSampler(
            sampling_strategy={ t: N_per_group for t in target_classes }, 
            random_state=seed, 
            replacement=False
        ) 
    elif balancing_method == "bootstrap":
        sampler = RandomUnderSampler(
            sampling_strategy={ t: N_per_group for t in target_classes }, 
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

def preprocess_grouped_dataset(X, y, ids, testset_ratio, seed):
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
    scaler = MinMaxScaler()
    if testset_ratio != 0:
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X_imputed, y, ids, test_size=testset_ratio, random_state=seed)
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    else:
        X_train, y_train, id_train = X_imputed, y, ids
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled, y_test, id_test = pd.DataFrame(), pd.Series(), pd.Series()

    return {
        "X_train": X_train_scaled, 
        "X_test": X_test_scaled, 
        "y_train": y_train, 
        "y_test": y_test, 
        "id_train": id_train.reset_index(drop=True), 
        "id_test": id_test.reset_index(drop=True)
    }

def feature_selection(X, y, model_name, no_pca, explained_ratio, feature_num, seed, nfold=5):
    '''
    Inputs:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series)   : Target variable (i.e., age).
    - model_name (str): A key that specify the model to use for feature selection.
    - explained_ratio (float): The desired variance to be explained by the selected features.

    Return:
    - (list): Names of selected features.
    '''
    logging.info(f"Selecting data features using {model_name} (explained_ratio={explained_ratio}) ...")

    # Estimate model coefficients with cross-validation:
    if model_name == "LassoCV":
        model = LassoCV(cv=nfold, random_state=seed)
        model.fit(X, y)
        sort_by = np.abs(model.coef_) # absolute value of the coefficients

    elif model_name == "RF":
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
        model.fit(X, y)
        sort_by = model.feature_importances_

    elif model_name == "XGBR":
        model = XGBRegressor(n_estimators=100, random_state=seed)
        model.fit(X, y)
        sort_by = model.feature_importances_

    ## Rank the features based on the coefficients:
    ranked_features = [
        f[0] for f in sorted(
            zip(X.columns, sort_by), key=lambda x: x[1], reverse=True
        )
    ]

    # Apply PCA to determine the number of features that explain the desired variance
    if not no_pca:
        pca = PCA()
        pca.fit(X[ranked_features], y)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Find the number of components that explain the desired variance
        feature_num = np.argmax(cumulative_variance >= explained_ratio) + 1

    # Select the top features based on the number of components
    ranked_selected_features = ranked_features[:feature_num]

    return list(ranked_selected_features)

def optimize_hyperparameters(trial, X, y, model_name, seed): 
    '''
    Inputs:
    - trial (optuna.trial.Trial)
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series)   : Target variable (i.e., age).
    - model_name (str): A key that specify the models to evaluate.

    Return:
    - (float): Average mean absolute error (MAE) score across cross validation.
    '''
    if model_name == "ElasticNet": 
        model = ElasticNet(
            alpha=trial.suggest_float('alpha', 1e-5, 1, log=True), 
            l1_ratio=trial.suggest_float('l1_ratio', 0, 1), 
            random_state=seed
        )
    elif model_name == "RF": 
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 10, 200), 
            max_depth=trial.suggest_int('max_depth', 2, 32), 
            random_state=seed
        )
    elif model_name == "CART": 
        model = DecisionTreeRegressor(
            max_depth=trial.suggest_int('max_depth', 2, 32), 
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20), 
            random_state=seed
        )
    elif model_name == "LGBM": 
        model = LGBMRegressor(
            objective='regression',
            metric='mae',
            num_leaves=trial.suggest_int('num_leaves', 2, 256),
            learning_rate=trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
            feature_fraction=trial.suggest_float('feature_fraction', 0.1, 1.0),
            bagging_fraction=trial.suggest_float('bagging_fraction', 0.1, 1.0),
            bagging_freq=trial.suggest_int('bagging_freq', 1, 7),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
            # device='cpu', # If GPU is not available
            random_state=seed
        )
    elif model_name == "XGBM": 
        model = XGBRegressor(
            max_depth=trial.suggest_int('max_depth', 1, 9),
            learning_rate=trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            subsample=trial.suggest_float('subsample', 0.1, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.1, 1.0),
            # tree_method='hist', # If GPU is not available
            random_state=seed
        )

    ## Evaluate the model using cross validation:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    mae_scores = []

    for n_fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Parameter optimization for {model_name}, trial {trial.number}, fold {n_fold+1} ...")

        if X.iloc[train_index].empty:
            return np.inf # If the training data is empty, return an error value
        else:
            model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = model.predict(X.iloc[val_index])
            mae_scores.append(mean_absolute_error(y.iloc[val_index], y_pred))

    return np.mean(mae_scores)

def train_and_evaluate(X, y, model_names, seed, optimize_trials=50):
    '''
    Inputs:
    - X (pd.DataFrame)  : Feature matrix.
    - y (pd.Series)     : Target variable (i.e., age).
    - model_names (list): List of keys that specify the models to evaluate.

    Return:
    - (dict): The evaluation results for each model, including:
        the mean and standard deviation of MAE scores, 
        and all included model with their best hyperparameters.
    '''
    results = {}

    for model_name in model_names:
        logging.info(f"Optimizing hyperparameters for {model_name} ...")

        ## Initialize the model with the best hyperparameters:
        study = optuna.create_study(
            direction='minimize', 
            sampler=optuna.samplers.RandomSampler(seed=seed)
        )
        study.optimize(
            lambda trial: optimize_hyperparameters(trial, X, y, model_name, seed), 
            n_trials=optimize_trials, 
            show_progress_bar=True
        )
        logging.info("Parameter optimization is completed :-)")

        if model_name == "ElasticNet":
            best_model = ElasticNet(**study.best_params, random_state=seed)
        elif model_name == "RF": 
            best_model = RandomForestRegressor(**study.best_params, random_state=seed)
        elif model_name == "CART": 
            best_model = DecisionTreeRegressor(**study.best_params, random_state=seed)
        elif model_name == "LGBM": 
            best_model = LGBMRegressor(**study.best_params, random_state=seed)
        elif model_name == "XGBM": 
            best_model = XGBRegressor(**study.best_params, random_state=seed)

        ## Calculate the mean and standard deviation of mean absolute error:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        mae_scores = []
        for n_fold, (train_index, val_index) in enumerate(kf.split(X)):
            logging.info(f"Evaluating {model_name}, fold {n_fold+1} ...")
            best_model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = best_model.predict(X.iloc[val_index])
            mae_scores.append(mean_absolute_error(y.iloc[val_index], y_pred))

        ## Storing results:
        results[model_name] = {
            "mae_mean"   : np.mean(mae_scores),
            "mae_std"    : np.std(mae_scores),
            "best_model" : best_model,
            "best_params": study.best_params 
        }

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

def convert_np_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert np.ndarray to list
    elif isinstance(obj, np.generic): 
        return obj.item()   # Convert np.generic to scalar
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
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed

    ## Define the type of the model to be used for training:
    if args.training_model is not None:
        included_models = [constant.model_names[args.training_model]]
        print_included_models = included_models
    elif args.pretrained_model_folder is not None:
        print_included_models = "Depend on the previous results"
    else:
        included_models = constant.model_names
        print_included_models = included_models

    ## Define the sampling method and number of participants per balanced group:
    if args.use_prepared_data: # should be balanced
        folder, _ = os.path.split(args.use_prepared_data)
        balancing_method, balancing_groups, N_per_group = "--", "--", "--" # default
        try:
            with open(os.path.join(folder, "description.json"), 'r', errors='ignore') as f:
                desc_json = json.load(f)
            balancing_method = desc_json["DataBalancingMethod"]
            balancing_groups = desc_json["BalancingGroups"]
            N_per_group = desc_json["NumPerBalancedGroup"]
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
            N_per_group = None

        if balancing_method is not None: # going to be balanced
            balancing_groups = config.balancing_groups
            if args.sample_size is None:
                N_per_group = constant.N_per_group[balancing_groups][balancing_method]
            else:
                N_per_group = args.sample_size

    ## Define the labels and boundaries of age groups:
    age_bin_labels = list(constant.age_groups[config.age_method].keys())
    age_boundaries = list(constant.age_groups[config.age_method].values()) 

    ## Define the labels and boundaries for age correction:
    pad_age_groups = list(constant.age_groups[config.pad_method].keys())
    pad_age_breaks = [ 0 ] + [ x for _, x in list(constant.age_groups[config.pad_method].values()) ] 

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

    ## Save the description of the current execution as a JSON file:
    desc = {
        "Seed": seed, 
        "UsePreparedData": args.use_prepared_data, 
        "RawDataVersion": config.data_file_path if args.use_prepared_data is None else "--", 
        "InclusionFileVersion": config.inclusion_file_path if args.use_prepared_data is None else "--", 
        "DataBalancingMethod": balancing_method, 
        "BalancingGroups": balancing_groups,
        "NumPerBalancedGroup": N_per_group, 
        "SexSeparated": config.by_gender, 
        "AgeGroups": age_bin_labels, 
        "TestsetRatio": config.testset_ratio, 
        "UsePretrainedModels": args.pretrained_model_folder, 
        "UsePreviouslySelectedFeatures": args.preselected_feature_folder, 
        "FeatureOrientations": list(constant.domain_approach_mapping.keys()),
        "FeatureSelectionModel": config.feature_selection_model, 
        "FeatureSelectionUsePCA": not args.no_pca,
        "FeatureExplainedRatio": config.explained_ratio, 
        "FeatureNum": constant.feature_num if not args.no_pca else None, 
        "IncludedOptimizationModels": print_included_models, 
        "SkippedIterationNum": args.ignore, 
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
    
    if args.use_prepared_data: # Use the prepared dataset
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
                N_per_group=N_per_group, 
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
            preprocessed_data_dicts[group_name] = preprocess_grouped_dataset(
                X=sub_DF.drop(columns=["ID", "BASIC_INFO_AGE"]), 
                y=sub_DF["BASIC_INFO_AGE"], 
                ids=sub_DF["ID"], 
                testset_ratio=config.testset_ratio, 
                seed=seed
            ) # a dictionary of train-test splited data, storing 
              # the standardized feature values, age, and ID numbers of participants.

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

                if iter < args.ignore: # Skip the current iteration
                    logging.info(f"Skipping {iter}-th iteration :-O")

                else: 
                    if args.pretrained_model_folder is not None:                       
                        logging.info("Using the pre-trained model :-P")

                        saved_json = os.path.join("outputs", args.pretrained_model_folder, f"results_{group_name}_{ori_name}.json")
                        with open(saved_json, 'r', encoding='utf-8') as f:
                            saved_results = json.load(f)

                        best_model_name = saved_results["Model"]
                        selected_features = saved_results["FeatureNames"]
                        mean_train_mae = saved_results["MeanTrainMAE"]

                        saved_model = os.path.join("outputs", args.pretrained_model_folder, f"models_{group_name}_{ori_name}_{best_model_name}.pkl")
                        with open(saved_model, 'rb') as f:
                            best_model = pickle.load(f)
                        
                        X_train_selected = data_dict["X_train"].loc[:, selected_features]

                    else: 
                        logging.info("Training model from scratch ...")

                        if args.preselected_feature_folder is not None:                        
                            logging.info("Using previously selected features :-P")

                            saved_json = os.path.join("outputs", args.preselected_feature_folder, f"results_{group_name}_{ori_name}.json")
                            with open(saved_json, 'r', errors='ignore') as f:
                                saved_results = json.load(f)

                            best_model_name = saved_results["Model"]
                            selected_features = saved_results["FeatureNames"]

                        else: 
                            logging.info("Selecting features ...")
                            included_features = [ # filter the features based on the domain and approach
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

                            else: 
                                selected_features = feature_selection(
                                    X=data_dict["X_train"].loc[:, included_features], 
                                    y=data_dict["y_train"], 
                                    model_name=config.feature_selection_model, 
                                    no_pca=args.no_pca, 
                                    explained_ratio=config.explained_ratio, 
                                    feature_num=constant.feature_num, 
                                    seed=seed
                                )

                        if len(selected_features) == 0: 
                            logging.warning("No features are selected for the current orientation!!")
                            logging.info("Unable to train models, skipping ...")
                            record_if_failed.append(f"{ori_name} of {group_name}.")
                            continue

                        else:
                            X_train_selected = data_dict["X_train"].loc[:, selected_features]                            

                            if X_train_selected.empty: 
                                logging.warning("No data left after feature selection!!")
                                logging.info("Unable to train models, skipping ...")
                                record_if_failed.append(f"{ori_name} of {group_name} after feature selection.")
                                continue

                            else:
                                if (args.training_model is None) and (args.preselected_feature_folder is not None):
                                    included_models = [best_model_name]
                                    logging.info(f"Since previously selected features are used, using the corresponding model type: {best_model_name}")
                                else:
                                    logging.info(f"Using best model from the following model types: {included_models}")
                                
                                logging.info("Training and evaluating models ...")
                                results = train_and_evaluate(
                                    X=X_train_selected, 
                                    y=data_dict["y_train"], 
                                    model_names=included_models, 
                                    seed=seed
                                ) # Including the mean and standard deviation of MAE scores, and all included model with their best hyperparameters.

                                best_model_name = min(results, key=lambda x: results[x]["mae_mean"])
                                best_model = results[best_model_name]["best_model"]
                                mean_train_mae = results[best_model_name]["mae_mean"]

                                logging.info("Saving the best model to the main output folder ...")
                                model_outpath = config.model_outpath_format.format(group_name, ori_name, best_model_name)
                                with open(model_outpath, 'wb') as f:
                                    pickle.dump(best_model, f)

                                logging.info("Saving other models to the 'other models' folder ...")
                                for model_name, model_result in results.items():
                                    if model_name != best_model_name: 
                                        fp, fn = os.path.split(config.model_outpath_format.format(group_name, ori_name, model_name))
                                        os.makedirs(os.path.join(fp, "other models"), exist_ok=True)
                                        model_outpath = os.path.join(fp, "other models", fn)
                                        with open(model_outpath, 'wb') as f:
                                            pickle.dump(model_result["best_model"], f)

                                logging.info("Saving the mean and std of MAE values for all models ...")   
                                model_maes = {
                                    model_name: {
                                        'Mean': res["mae_mean"], 
                                        'STD': res["mae_std"]
                                    } for model_name, res in results.items()
                                }
                                model_maes_outpath = config.model_maes_outpath_format.format(group_name, ori_name)
                                with open(model_maes_outpath, 'w') as f:
                                    json.dump(model_maes, f)
                                
                    logging.info("Applying the best model to the training set ...")
                    y_pred_train = best_model.predict(X_train_selected)
                    pad_train = y_pred_train - data_dict["y_train"]

                    logging.info("Generating age-correction reference table ...")
                    correction_ref = generate_correction_ref(
                        age=data_dict["y_train"], 
                        pad=pad_train, 
                        age_groups=pad_age_groups, 
                        age_breaks=pad_age_breaks
                    )
                            
                    if data_dict["X_test"].empty: 
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

                        save_results = {
                            "Model": best_model_name, 
                            "MeanTrainMAE": mean_train_mae, 
                            "NumberOfSubjs": len(data_dict["id_train"]), 
                            "SubjID": list(data_dict["id_train"]), 
                            "Note": "Train and test sets are the same.", 
                            "Age": list(data_dict["y_train"]), 
                            "PredictedAge": list(y_pred_train), 
                            "PredictedAgeDifference": list(pad_train), 
                            "CorrectedPAD": list(padac_train), 
                            "CorrectedPredictedAge": list(corrected_y_pred_train), 
                            "AgeCorrectionTable": correction_ref.to_dict(orient='records'), 
                            "NumberOfFeatures": len(selected_features), 
                            "FeatureNames": selected_features, 
                        }

                    else:
                        logging.info("Evaluating the best model on the testing set ...")
                        X_test_selected = data_dict["X_test"].loc[:, selected_features]                       
                        y_pred_test = best_model.predict(X_test_selected)
                        y_pred_test = pd.Series(y_pred_test, index=data_dict["y_test"].index)
                        pad = y_pred_test - data_dict["y_test"]

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

                        save_results = {
                            "Model": best_model_name, 
                            "MeanTrainMAE": mean_train_mae, 
                            "NumberOfTraining": len(data_dict["id_train"]), 
                            "TrainingSubjID": list(data_dict["id_train"]), 
                            "NumberOfTesting": len(data_dict["id_test"]), 
                            "TestingSubjID": list(data_dict["id_test"]), 
                            "Age": list(data_dict["y_test"]), 
                            "PredictedAge": list(y_pred_test), 
                            "PredictedAgeDifference": list(pad), 
                            "CorrectedPAD": list(padac), 
                            "CorrectedPredictedAge": list(corrected_y_pred_test), 
                            "AgeCorrectionTable": correction_ref.to_dict(orient='records'), 
                            "NumberOfFeatures": len(selected_features), 
                            "FeatureNames": selected_features, 
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


