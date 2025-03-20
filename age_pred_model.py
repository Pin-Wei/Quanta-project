#!/usr/bin/python

import os
import argparse
import logging
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import copy
from imblearn.over_sampling import SMOTE # ADASYN 
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
parser.add_argument("-age", "--age_method", type=int, default=0, 
                    help="The method to define age groups. Options: 0 (cut_at_40), 1 (wais_8_seg).")
parser.add_argument("-sex", "--sep_sex", action="store_true", default=False, 
                    help="Whether to separate the data by gender. (default: False).")
## Balancing data such that all groups have the same number of participants:
parser.add_argument("-u", "--upsample", action="store_true", default=False, 
                    help="Up-sample the data using SMOTE.")
parser.add_argument("-d", "--downsample", action="store_true", default=False, 
                    help="Down-sample the data without replacement.")
parser.add_argument("-b", "--bootstrap", action="store_true", default=False, 
                    help="Down-sample the data with replacement (i.e., bootstrapping).")
parser.add_argument("-n", "--sample_size", type=int, default=None, 
                    help="The number of participants to up- or down-sample to.")
## Split data into training and testing sets:
parser.add_argument("-tsr", "--testset_ratio", type=float, default=0.0, 
                    help="The ratio of the testing set.")
## Feature selection:
parser.add_argument("-wda", "--without_domain_approach", action="store_true", default=False, 
                    help="Whether not to use domain approach for feature selection. (default: False).")
parser.add_argument("-fsm", "--feature_selection_model", type=int, default=0, 
                    help="The model to use for feature selection. Options: 0 (LassoCV), 1 (RF), 2 (XGBR).")
parser.add_argument("-epr", "--explained_ratio", type=float, default=0.9, 
                    help="The variance to be explained by the selected features.")
## Model training:
parser.add_argument("-pmf", "--pretrained_model_folder", type=str, default=None, 
                    help="Folder containing the pre-trained model files (.pkl).")
parser.add_argument("-i", "--ignore", type=int, default=0, 
                    help="Ignore the first N iterations (In case you might be interrupted and don't want to start from the beginning)ㄡ")
parser.add_argument("-s", "--seed", type=int, default=None, 
                    help="The value used to initialize all random number generator.")
args = parser.parse_args()

## Classes: ===========================================================================

class Config:
    def __init__(self):
        self.data_file_path = os.path.join("rawdata", "DATA_ses-01_2024-12-09.csv")
        self.inclusion_file_path = os.path.join("rawdata", "InclusionList_ses-01.csv")
        self.age_method = ["cut_at_40", "wais_8_seg"][args.age_method]
        self.sep_sex = args.sep_sex
        self.testset_ratio = args.testset_ratio
        self.feature_selection_model = ["LassoCV", "RF", "XGBR"][args.feature_selection_model]
        self.explained_ratio = args.explained_ratio        
        self.pad_method = ["wais_8_seg", "every_5_yrs"][0]
        self.out_folder = os.path.join("outputs", datetime.today().strftime('%Y-%m-%d_%H.%M.%S'))
        self.description_outpath = os.path.join(self.out_folder, "description.json")
        self.balanced_data_outpath = os.path.join(self.out_folder, "balanced_data.csv")
        self.logging_outpath = os.path.join(self.out_folder, "log.txt")
        self.failure_record_outpath = os.path.join(self.out_folder, "failure_record.txt")
        self.results_outpath_template = os.path.join(self.out_folder, "results_groupname_oriname.json")
        self.model_outpath_template = os.path.join(self.out_folder, "models_groupname_oriname_modeltype.pkl")

class Constants:
    def __init__(self):
        ## The age groups defined by different methods:
        self.age_groups = { 
            "cut_at_40": {
                "le-40" : ( 0, 40),    # less than or equal to
                "ge-41" : (41, np.inf) # greater than or equal to
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
            "SMOTE": 60, 
            "downsample": 15, 
            "bootstrap": 15
        }

## Functions: =========================================================================

def load_and_merge_datasets(data_file_path, inclusion_file_path):
    '''
    Read the data and inclusion table, and merge them.
    '''
    ## Load the main dataset:
    DF = pd.read_csv(data_file_path)
    logging.info("Successfully loaded the main dataset.")
    
    ## Load the file marking whether a data has been collected from individual participants:
    inclusion_df = pd.read_csv(inclusion_file_path)
    logging.info("Successfully loaded the inclusion table.")

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
    
    OverSampling using 'SMOTE' (Synthetic Minority Oversampling Technique)
    - see: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE.fit_resample
    
    UnderSampling using 'RandomUnderSampler' with (bootstrap) or without replacement (downsample)
    - see: https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html#imblearn.under_sampling.RandomUnderSampler
    '''
    ## Assign "AGE-GROUP_SEX" labels:
    DF["AGE-GROUP"] = pd.cut(
        x=DF["BASIC_INFO_AGE"], 
        bins=[ x for x, _ in list(age_bin_dict.values()) ] + [ np.inf ], 
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
    # N_per_group = 0 if N_per_group is None else N_per_group
    for t in target_classes:
        sub_DF = DF[DF[target_col] == t]
        sub_X = sub_DF.drop(columns=[target_col])
        imputer = SimpleImputer(strategy="median")
        sub_DF_imputed = pd.DataFrame(imputer.fit_transform(sub_X), columns=sub_X.columns)
        sub_DF_imputed[target_col] = sub_DF[target_col].reset_index(drop=True)
        DF_imputed_list.append(sub_DF_imputed)
        # if (balancing_method == "SMOTE") and (len(sub_DF_imputed) > N_per_group): 
        #     N_per_group = len(sub_DF_imputed)
        # elif (balancing_method == "sampling" | balancing_method == "bootstrap") and (len(sub_DF_imputed) < N_per_group):
        #     N_per_group = len(sub_DF_imputed)
    DF_imputed = pd.concat(DF_imputed_list)

    ## Make balanced datasets:    
    if balancing_method == "SMOTE":
        sampler = SMOTE(
            sampling_strategy={ t: N_per_group for t in target_classes }, 
            random_state=seed
        )
    elif balancing_method == "sampling":
        sampler = RandomUnderSampler(
            sampling_strategy={ t: N_per_group for t in target_classes }, 
            random_state=seed
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
    DF_balanced["ID"] = [ f"sub-{x:04d}" for x in DF_balanced.index ]

    return target_col, DF_balanced

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
            X, y, ids, test_size=testset_ratio, random_state=seed)
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

def feature_selection(X, y, model_name, explained_ratio, seed, nfold=5):
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
    pca = PCA()
    pca.fit(X[ranked_features], y)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components that explain the desired variance
    num_components = np.argmax(cumulative_variance >= explained_ratio) + 1

    # Select the top features based on the number of components
    ranked_selected_features = ranked_features[:num_components]

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
        logging.info(f"Parameter optimization for {model_name}, trial {trial.number}, fold {n_fold+1} ...")

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
        the best model, and the best hyperparameters.
    '''
    results = {}

    for model_name in model_names:

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
        logging.info(f"Parameter optimization for {model_name} is completed.")

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

    ## Create output folder if it doesn't exist:
    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)

    ## Setup logging file:
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

    ## Define the sampling method and number of participants per group:
    if args.upsample:
        balancing_method = "SMOTE"
    elif args.downsample:
        balancing_method = "downsample"
    elif args.bootstrap:
        balancing_method = "bootstrap"
    else:
        balancing_method = None
        N_per_group = None

    if balancing_method is not None:
        if args.sample_size is None:
            N_per_group = constant.N_per_group[balancing_method]
        else:
            N_per_group = args.sample_size

    ## Define the labels and boundaries of age groups:
    age_bin_labels = list(constant.age_groups[config.age_method].keys())
    age_boundaries = list(constant.age_groups[config.age_method].values()) 

    ## Define the labels and boundaries for age correction:
    pad_age_groups = list(constant.age_groups[config.pad_method].keys())
    pad_age_breaks = [ x for x, _ in list(constant.age_groups[config.pad_method].values()) ] + [ np.inf ]

    ## Revise the mapping to include all domains and approachs, if specified:
    if args.without_domain_approach:
        logging.info("Domain approach is not used for feature selection.")
        constant.domain_approach_mapping = {
            "ALL": {
                "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
                "approaches": ["MRI", "BEH", "EEG"]
            }
        }

    ## Save the description of the current execution as a JSON file:
    desc = {
        "DataVersion": config.data_file_path, 
        "InclusionVersion": config.inclusion_file_path, 
        "DataBalancingMethod": balancing_method, 
        "NumPerGroup": N_per_group,
        "Seed": seed, 
        "SexSeparated": config.sep_sex, 
        "AgeGroups": age_bin_labels, 
        "IgnoreFirstGroups": args.ignore, 
        "CorrectionAgeGroups": pad_age_groups, 
        "TestsetRatio": config.testset_ratio, 
        "FeatureOrientations": list(constant.domain_approach_mapping.keys()),
        "FeatureSelectionModel": config.feature_selection_model, 
        "FeatureExplainedRatio": config.explained_ratio, 
        "OptimizedModels": constant.model_names, 
        "UsePretrainedModels": args.pretrained_model_folder, 
        "WithoutDomainApproach": args.without_domain_approach
    }
    desc = convert_np_types(desc)

    with open(config.description_outpath, 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False)

    logging.info("The description of the current execution has been saved as a JSON file.")

    ## Copy the current Python script to the output folder:
    shutil.copyfile(
        src=os.path.abspath(__file__), 
        dst=os.path.join(config.out_folder, os.path.basename(__file__))
    )

    logging.info("The current Python script has been copied to the output folder.")

    ## Record the failed processing to a text file:
    record_if_failed = [] 
    
## STEP-1. Load data, split into groups, and preprocess -------------------------------------------------------

    ## Load the raw dataset:
    DF = load_and_merge_datasets(
        data_file_path=config.data_file_path, 
        inclusion_file_path=config.inclusion_file_path
    )

    ## Make balanced datasets if specified:
    if balancing_method is not None:
        target_col, DF_balanced = make_balanced_dataset(
            DF=copy.deepcopy(DF), 
            balancing_method=balancing_method, 
            age_bin_dict=constant.age_groups["wais_8_seg"], 
            N_per_group=N_per_group, 
            seed=seed
        )
        DF_balanced.to_csv(config.balanced_data_outpath, index=False)
        DF_balanced.drop(columns=[target_col], inplace=True)
    else:
        DF_balanced = DF

    ## Divide the dataset into groups and define their labels:
    if args.sep_sex:
        logging.info("Separating data according to participants' age ranges and genders.")
        sub_DF_list = [
            DF_balanced[(DF_balanced["BASIC_INFO_AGE"].between(lb, ub)) & (DF_balanced["BASIC_INFO_SEX"] == sex)] 
            for (lb, ub), sex in list(product(age_boundaries, [1, 2]))
        ]
        sub_DF_labels = [ f"{age_group}_{sex}" for age_group, sex in list(product(age_bin_labels, ["M", "F"])) ] 
    else:
        logging.info("Separating data according to participants' age ranges.")
        sub_DF_list = [
            DF_balanced[DF_balanced["BASIC_INFO_AGE"].between(lb, ub)] 
            for lb, ub in age_boundaries
        ]
        sub_DF_labels = age_bin_labels

    ## Separately preprocess different data subsets: 
    preprocessed_data_dicts = {}

    for group_name, sub_DF in zip(sub_DF_labels, sub_DF_list):

        if sub_DF.empty:
            preprocessed_data_dicts[group_name] = None
        else:
            sub_DF = sub_DF.reset_index(drop=True)
            preprocessed_data_dicts[group_name] = preprocess_grouped_dataset(
                X=sub_DF.drop(columns=["ID", "BASIC_INFO_AGE"]), 
                y=sub_DF["BASIC_INFO_AGE"], 
                ids=sub_DF["ID"], 
                testset_ratio=config.testset_ratio, 
                seed=seed
            ) # a dictionary of train-test splited data, storing 
              # the standardized feature values, age, and ID numbers of participants.

## STEP-2. Feature selection -----------------------------------------------------------------------------------

    iter = 0 # iteration counter for skipping

    for group_name, data_dict in preprocessed_data_dicts.items():

        if data_dict is None: 
            logging.warning(f"Unable to process data for group '{group_name}'.")
            record_if_failed.append(f"Entire {group_name}.")
            continue

        else: 
            for ori_name, ori_content in constant.domain_approach_mapping.items():

                ### Skip the current iteration
                if iter < args.ignore: 
                    logging.info(f"Skipping iteration: group='{group_name}', type='{ori_name}'")
                    iter += 1

                ### Continue the current iteration
                else: 

                    #### Train models from scratch:
                    if args.pretrained_model_folder is None: 
                        logging.info(f"Processing group: {group_name}, type: {ori_name}")
                        iter += 1

                        included_features = [ # filter the features based on the domain and approach
                            col for col in data_dict["X_train"].columns
                            if any( domain in col for domain in ori_content["domains"] )
                            and any( app in col for app in ori_content["approaches"] )
                            and "RESTING" not in col
                        ]

                        if ori_name == "FUNCTIONAL": # exclude "STRUCTURE" features
                            included_features = [ col for col in included_features if "STRUCTURE" not in col ]
                        
                        if len(included_features) == 0: 
                            logging.warning(f"There are no available features for orientation '{ori_name}' in group '{group_name}'.")
                            record_if_failed.append(f"{ori_name} of {group_name}.")
                            continue

                        else: 
                            selected_features = feature_selection(
                                X=data_dict["X_train"].loc[:, included_features], 
                                y=data_dict["y_train"], 
                                model_name=config.feature_selection_model, 
                                explained_ratio=config.explained_ratio, 
                                seed=seed
                            )

## STEP-3. Find the best model and save its parameters -------------------------------------------------------

                            X_train_selected = data_dict["X_train"].loc[:, selected_features]

                            if X_train_selected.empty: 
                                logging.warning(f"After feature selection, there are no available features for orientation '{ori_name}' in group '{group_name}'.")
                                record_if_failed.append(f"{ori_name} of {group_name} after feature selection.")
                                continue

                            else:                    
                                results = train_and_evaluate(
                                    X=X_train_selected, 
                                    y=data_dict["y_train"], 
                                    model_names=constant.model_names, 
                                    seed=seed
                                ) # Including ...
                                    # the mean and standard deviation of MAE scores, ...
                                    # the best model, and the best hyperparameters.

                                best_model_name = min(results, key=lambda x: results[x]["mae_mean"])
                                best_model = results[best_model_name]["best_model"]
                                mean_train_mae = results[best_model_name]["mae_mean"]

                                model_outpath = (
                                    config.model_outpath_template
                                    .replace("groupname", group_name)
                                    .replace("oriname", ori_name)
                                    .replace("modeltype", best_model_name)
                                )
                                with open(model_outpath, 'wb') as f:
                                    pickle.dump(best_model, f)

                                logging.info(f"The trained model have been saved for group '{group_name}' and orientation '{ori_name}'.")

                    #### Use the pre-trained model:
                    else: 
                        logging.info(f"Using pre-trained models for group '{group_name}' and orientation '{ori_name}'.")
                        previous_path = os.path.join("outputs", args.pretrained_model_folder)
                        
                        saved_json = f"results_{group_name}_{ori_name}.json"
                        with open(os.path.join(previous_path, saved_json), 'r', encoding='utf-8') as f:
                            saved_results = json.load(f)
                            best_model_name = saved_results["Model"]
                            selected_features = saved_results["FeatureNames"]
                            mean_train_mae = saved_results["MeanTrainMAE"]

                        saved_model = f"models_{group_name}_{ori_name}_{best_model_name}.pkl"
                        with open(os.path.join(previous_path, saved_model), 'rb') as f:
                            best_model = pickle.load(f)
                        
                        X_train_selected = data_dict["X_train"].loc[:, selected_features]
                        
## STEP-4. Generate age-correction reference table -----------------------------------------------------------

                    y_pred_train = best_model.predict(X_train_selected)
                    pad_train = y_pred_train - data_dict["y_train"]
                            
                    correction_ref = generate_correction_ref(
                        age=data_dict["y_train"], 
                        pad=pad_train, 
                        age_groups=pad_age_groups, 
                        age_breaks=pad_age_breaks
                    )

## STEP-5. Apply the model, apply age-correction, and save the results ----------------------------------------
                            
                    if data_dict["X_test"].empty: # apply the model to the training set
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
                        X_test_selected = data_dict["X_test"].loc[:, selected_features]                       
                        y_pred_test = best_model.predict(X_test_selected)
                        y_pred_test = pd.Series(y_pred_test, index=data_dict["y_test"].index)
                        pad = y_pred_test - data_dict["y_test"]
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
                    fp1 = (config.results_outpath_template
                        .replace("groupname", group_name)
                        .replace("oriname", ori_name))
                    with open(fp1, 'w', encoding='utf-8') as f:
                        json.dump(save_results, f, ensure_ascii=False)

                    logging.info(f"Model prediction have been saved as JSON files for group '{group_name}' and orientation '{ori_name}'.")

                ### Next iteration

    with open(config.failure_record_outpath, 'w') as f:
        f.write("\n".join(record_if_failed))
    
    logging.info("The record of failed processing have been saved.")

if __name__ == "__main__":
    main()


