#!/usr/bin/python

import os
import sys
import json
import pickle
import shutil
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

import optuna
import optunahub
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, ElasticNetCV, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.join(os.getcwd(), "..", "src"))
from data_preparing import DataManager
from feature_engineering import FeatureReducer, FeatureSelector

## Classes: ===========================================================================

class Constants:
    def __init__(self, args):
        ## The age groups defined by different methods:
        self.age_groups = { 
            "no-cut": {
                "all": (0, np.inf)
            }, 
            "cut-40-41": {
                "le-40" : ( 0, 40),    # less than or equal to
                "ge-41" : (41, np.inf) # greater than or equal to
            }, 
            "cut-44-45": {
                "le-44" : ( 0, 44),
                "ge-45" : (45, np.inf) 
            }, 
            "cut-8-wais": {
                "le-24": ( 0, 24), 
                "25-29": (25, 29), 
                "30-34": (30, 34),
                "35-44": (35, 44), 
                "45-54": (45, 54), 
                "55-64": (55, 64), 
                "65-69": (65, 69), 
                "ge-70": (70, np.inf)
            }, 
            "every-5-yrs": {
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
        self.all_domain_approach_mapping = {
            "ALL": {
                "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
                "approaches": ["MRI", "BEH", "EEG"]
            }
        }
        if args.include_all_mappings:
            self.domain_approach_mapping.update(self.all_domain_approach_mapping)
        elif args.only_all_mapping:
            self.domain_approach_mapping = self.all_domain_approach_mapping

        ## The names of models to evaluate:
        self.model_names = [ 
            "ElasticNet", 
            "RF",   # RandomForestRegressor
            "CART", # DecisionTreeRegressor
            "LGBM", # lgb.LGBMRegressor
            "XGBM"  # xgb.XGBRegressor
        ]

        ## The number of participants in each balanced group:
        self.balance_to_num = {
            "cut-8-wais": {
                "CTGAN": 60, 
                "TVAE": 60,
                "SMOTENC": 60, 
                "downsample": 15, 
                "bootstrap": 15
            }, 
            "cut-44-45": {
                "CTGAN": 60*4, 
                "TVAE": 60*4, 
                "SMOTENC": 60*4, 
                "downsample": 15*4, 
                "bootstrap": 15*4
            }
        }
        ## Number of parallel threads to use:
        self.n_jobs = 16

class Config:
    def __init__(self, args, constants):
        self.source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.raw_data_path = os.path.join(self.source_path, "data", "rawdata", "DATA_ses-01_2024-12-09.csv")
        self.inclusion_data_path = os.path.join(self.source_path, "data", "rawdata", "InclusionList_ses-01.csv")
        self.syndata_folder = os.path.join(self.source_path, "data", "syndata")
        self = self._setup_vars(args, constants)
        self = self._setup_out_folder(args)
        self.logging_outpath                 = os.path.join(self.out_folder, "log.txt")
        self.description_outpath             = os.path.join(self.out_folder, "description.json")
        self.prepared_data_outpath           = os.path.join(self.out_folder, "prepared_data.csv")
        self.split_ids_outpath               = os.path.join(self.out_folder, "train_and_test_ids.json")
        self.scaler_outpath_format           = os.path.join(self.out_folder, "scaler_{}.pkl")
        self.reducer_outpath_format          = os.path.join(self.out_folder, "reducer_{}_{}.pkl")
        self.pc_loadings_outpath_format      = os.path.join(self.out_folder, "PC_loadings_{}_{}.xlsx")
        self.rc_loadings_outpath_format      = os.path.join(self.out_folder, "RC_loadings_{}_{}.xlsx")
        self.dropped_features_outpath_format = os.path.join(self.out_folder, "dropped_features_{}_{}.txt")
        self.model_outpath_format            = os.path.join(self.out_folder, "models_{}_{}_{}.pkl")
        self.features_outpath_format         = os.path.join(self.out_folder, "features_{}_{}.csv")
        self.training_results_outpath_format = os.path.join(self.out_folder, "training_results_{}_{}.json")
        self.results_outpath_format          = os.path.join(self.out_folder, "results_{}_{}.json")
        self.failure_record_outpath          = os.path.join(self.out_folder, "failure_record.txt")

    def _setup_vars(self, args, constants):
        self.seed = args.seed if args.seed is not None else np.random.randint(0, 10000)
        self.by_gender = [False, True][args.by_gender]
        self.age_method = ["no-cut", "cut-40-41", "cut-44-45", "cut-8-wais"][args.age_method]
        self.age_bin_labels = list(constants.age_groups[self.age_method].keys())
        self.age_boundaries = list(constants.age_groups[self.age_method].values())
        self.age_correction_method = ["Zhang et al. (2023)", "Beheshti et al. (2019)"][args.age_correction_method]
        self.age_correction_groups = ["cut-8-wais", "every-5-yrs"][0]
        self.pad_age_groups = list(constants.age_groups[self.age_correction_groups].keys())
        self.pad_age_breaks = [ 0 ] + [ x for _, x in list(constants.age_groups[self.age_correction_groups].values()) ] 
        # self.feature_selection_method = [None, "LassoCV", "ElasticNetCV", "RF-Permute", "ElaNet-SHAP", "RF-SHAP", "LGBM-SHAP"][args.feature_selection_method]
        # self.fs_thresh_method = ["fixed_threshold", "explained_ratio"][args.fs_thresh_method]

        self.balancing_method = [None, "CTGAN", "TVAE", "SMOTENC", "downsample", "bootstrap"][args.balancing_method]
        self.balancing_groups = ["cut-8-wais", "cut-44-45"][args.balancing_groups]
        self.balance_to_num = None
        if self.balancing_method is not None:
            if args.sample_size is None:
                self.balance_to_num = constants.balance_to_num[self.balancing_groups][self.balancing_method]
            else:
                self.balance_to_num = args.sample_size
        
        if args.split_with_ids is not None:
            with open(args.split_with_ids, 'r') as f:
                self.split_with_ids = json.load(f)
            self.testset_ratio = len(self.split_with_ids["Test"]) / (len(self.split_with_ids["Train"]) + len(self.split_with_ids["Test"]))
        else:
            self.split_with_ids = None
            self.testset_ratio = args.testset_ratio

        if args.training_model is not None:
            self.included_models = [ constants.model_names[args.training_model] ]
        else:
            self.included_models = constants.model_names

        return self

    def _setup_out_folder(self, args):
        prefix = datetime.today().strftime('%Y-%m-%d')

        if args.use_prepared_data:
            balancing_method = os.path.basename(os.path.dirname(args.use_prepared_data)).split("_")[0]
            prefix += f"_{balancing_method.lower()}"
        elif self.balancing_method is not None:
            prefix += f"_{self.balancing_method.lower()}"
        else:
            prefix += "_original"

        if args.age_method == 0:
            prefix += "_age-0"

        if args.by_gender == 0:
            prefix += "_sex-0"

        if args.pretrained_model_folder is not None:
            self.out_folder = os.path.join(self.source_path, "outputs", f"{prefix}_pre-trained")
        else:
            self.out_folder = os.path.join(self.source_path, "outputs", prefix)

        while os.path.exists(self.out_folder): # make sure the output folder does not exist:
            self.out_folder = self.out_folder + "+"

        return self

## Functions: =========================================================================

def define_arguments():
    parser = argparse.ArgumentParser(description="")

    ## Data preparation:
    parser.add_argument("-dat", "--prepare_data_and_exit", action="store_true", default=False, 
                        help="Only make the data, do not train any model.")
    parser.add_argument("-upd", "--use_prepared_data", type=str, default=None, 
                        help="File path of the data to be used (.csv).")
    
    ## Balancing data such that all groups have the same number of participants:
    parser.add_argument("-bm", "--balancing_method", type=int, default=0, 
                        help="The method to balance the data (0: None, 1: CTGAN, 2: TVAE, 3: SMOTENC, 4: downsample, 5: bootstrap).")
    parser.add_argument("-bg", "--balancing_groups", type=int, default=0, 
                        help="The groups to be balanced.")
    parser.add_argument("-n", "--sample_size", type=int, default=None, 
                        help="The number of participants to up- or down-sample to. (if None, use the default number of participants per group set in constants.balance_to_num).")
    
    ## How to separate data into groups:
    parser.add_argument("-age", "--age_method", type=int, default=2, 
                        help="The method to define age groups (0: 'no-cut', 1: 'cut-40-41', 2: 'cut-44-45', 3: 'cut-8-wais').")
    parser.add_argument("-sex", "--by_gender", type=int, default=0, 
                        help="Whether to separate the data by gender (0: False, 1: True).")
    
    ## Split data into training and testing sets:
    parser.add_argument("-tsr", "--testset_ratio", type=float, default=0.3, 
                        help="The ratio of the testing set.")
    parser.add_argument("-sid", "--split_with_ids", type=str, default=None, 
                        help="File path of a dictionary (.json) containing two lists of IDs to be used for splitting the data into training and testing sets. "+
                        "If provided, the testset_ratio will be determined by its length.")
    
    ## Feature selection:
    parser.add_argument("-iam", "--include_all_mappings", action="store_true", default=False, 
                        help="Include 'All' domain-approach mappings for feature selection.")
    parser.add_argument("-oam", "--only_all_mapping", action="store_true", default=False, 
                        help="Include only 'All' domain-approach mappings for feature selection.")
    parser.add_argument("-psf", "--preselected_feature_folder", type=str, default=None, 
                        help="The folder where the result files (.json) containing the selected features are stored.")
    # parser.add_argument("-fsm", "--feature_selection_method", type=int, default=2, 
    #                     help="The method to select features.")
    # parser.add_argument("-fst", "--fs_thresh_method", type=int, default=1, 
    #                     help="The method to determine the threshold for feature selection (0: 'threshold', 1: 'explained_ratio').")
    # parser.add_argument("-mfn", "--max_feature_num", type=int, default=None, 
    #                     help="The maximum number of features to be selected.")
    parser.add_argument("--no_pca", action="store_true", default=False, 
                        help="Do not use PCA for feature reduction.")
    parser.add_argument("-rhcf", "--remove_highly_correlated_features", action="store_true", default=False, 
                        help="Remove highly correlated features before feature reduction (PCA).")
    
    ## Model training:
    parser.add_argument("-pmf", "--pretrained_model_folder", type=str, default=None, 
                        help="The folder where the pre-trained model files (.pkl) are stored.")
    parser.add_argument("-m", "--training_model", type=int, default=None, 
                        help="The type of the model to be used for training (0: 'ElasticNet', 1: 'RF', 2: 'CART', 3: 'LGBM', 4: 'XGBM').")
    parser.add_argument("-i", "--ignore", type=int, default=0, 
                        help="Ignore the first N iterations (in case the script was interrupted by an accident and you don't want to start from the beginning).")
    parser.add_argument("-s", "--seed", type=int, default=9865, 
                        help="The value used to initialize all random number generator.")
    
    ## Age correction:
    parser.add_argument("-acm", "--age_correction_method", type=int, default=0, 
                        help="The method to correct age (0: 'Zhang et al. (2023)', 1: 'Beheshti et al. (2019)').")
    
    return parser.parse_args()

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
    
def save_description(args, config, constants):
    '''
    Save the description of the current execution as a JSON file.
    '''
    desc = {
        "Seed": config.seed, 
        "UsePreparedData": args.use_prepared_data, 
        "RawDataVersion": config.raw_data_path if args.use_prepared_data is None else "--", 
        "InclusionFileVersion": config.inclusion_data_path if args.use_prepared_data is None else "--", 
        "DataBalancingMethod": config.balancing_method, 
        "BalancingGroups": config.balancing_groups, 
        "NumPerBalancedGroup": config.balance_to_num
    }

    if args.use_prepared_data: 
        file_dir, _ = os.path.split(args.use_prepared_data)
        preceding_dir, folder = os.path.split(file_dir)
        if preceding_dir == config.syndata_folder:
            balancing_method, balancing_groups, balance_to_num = folder.split("_")[:3]
            desc["DataBalancingMethod"] = balancing_method
            desc["BalancingGroups"] = balancing_groups
            desc["NumPerBalancedGroup"] = balance_to_num
        else:
            try:
                with open(os.path.join(file_dir, "description.json"), 'r', errors='ignore') as f:
                    desc_json = json.load(f)
                desc["DataBalancingMethod"] = desc_json["DataBalancingMethod"]
                desc["BalancingGroups"] = desc_json["BalancingGroups"]
                desc["NumPerBalancedGroup"] = desc_json["NumPerBalancedGroup"]
            except:
                pass # no description file

    if not args.prepare_data_and_exit: 
        desc.update({
            "SexSeparated": config.by_gender, 
            "AgeGroups": config.age_bin_labels, 
            "UsePredefinedSplit": config.split_with_ids, 
            "TestsetRatio": config.testset_ratio, 
            "UsePretrainedModels": args.pretrained_model_folder, 
            "UsePreviouslySelectedFeatures": args.preselected_feature_folder, 
            "FeatureOrientations": list(constants.domain_approach_mapping.keys()), 
            # "FeatureSelectionMethod": config.feature_selection_method, 
            # "FSThresholdMethod": config.fs_thresh_method,
            # "MaxFeatureNum": args.max_feature_num, 
            "FeatureReductionMethod": "PCA" if not args.no_pca else "None",
            "RemoveHighlyCorrelatedFeatures": args.remove_highly_correlated_features, 
            "IncludedOptimizationModels": config.included_models if args.pretrained_model_folder is None else "Depend on the previous results", 
            "SkippedIterationNum": args.ignore,  
            "AgeCorrectionMethod": config.age_correction_method, 
            "AgeCorrectionGroups": config.pad_age_groups
        })

    ## Save the description to a JSON file:
    desc = convert_np_types(desc)
    with open(config.description_outpath, 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False)

    logging.info("The description of the current execution is saved :-)")
    
def divide_dataset(DF, config, divide_by_gender):
    '''
    Divide the dataset into groups based on participants' age (and gender).
    '''
    if divide_by_gender:
        group_name_list = [ f"{age_group}_{sex}" for age_group, sex 
            in list(product(config.age_bin_labels, ["M", "F"])) 
        ] 
        sub_DF_list = [
            DF[(DF["BASIC_INFO_AGE"].between(lb, ub)) & (DF["BASIC_INFO_SEX"] == sex)].reset_index(drop=True) 
            for (lb, ub), sex in list(product(config.age_boundaries, [1, 2]))
        ]
    else:
        group_name_list = config.age_bin_labels
        sub_DF_list = [
            DF[DF["BASIC_INFO_AGE"].between(lb, ub)].reset_index(drop=True) 
            for lb, ub in config.age_boundaries
        ]

    return group_name_list, sub_DF_list

def preprocess_divided_dataset(X, y, ids, split_with_ids, testset_ratio, trained_scaler, seed):
    '''
    Fill missing values, split into training and testing sets, 
    and perform feature scaling on the divided dataset. 
    '''
    def _remove_outlier_features(X):
        '''
        Remove features (data columns) with too many missing observations.
        '''
        n_subjs = len(X)
        na_rates = pd.Series(X.isnull().sum() / n_subjs)
        Q1 = na_rates.quantile(.25)
        Q3 = na_rates.quantile(.75)
        IQR = Q3 - Q1
        outliers = na_rates[na_rates > (Q3 + IQR * 1.5)]

        return X.drop(columns=outliers.index)

    def _get_normed_train_test(X, y, ids, split_with_ids, testset_ratio, trained_scaler, seed):
        if trained_scaler is None:
            scaler = StandardScaler() # to have zero mean and unit variance
            trained_or_not = "not_trained"
        else:
            scaler = trained_scaler
            trained_or_not = "trained"

        if testset_ratio == 0: # no testing set, use the whole dataset for training
            X_train_scaled = _feature_scaling(X, scaler, trained_or_not)
            y_train, id_train = y, ids
            X_test_scaled, y_test, id_test = pd.DataFrame(), pd.Series(), pd.Series()

        else: 
            if split_with_ids is not None: # split the dataset based on pre-defined IDs
                idx_train, idx_test = ids.isin(split_with_ids["Train"]), ids.isin(split_with_ids["Test"])
                X_train, y_train, id_train = X[idx_train], y[idx_train], ids[idx_train]
                X_test, y_test, id_test = X[idx_test], y[idx_test], ids[idx_test]
            
            else: # split the dataset based on the given ratio
                X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                    X, y, ids, test_size=testset_ratio, random_state=seed
                )

            X_train_scaled = _feature_scaling(X_train, scaler, trained_or_not)
            X_test_scaled = _feature_scaling(X_test, scaler, "trained")

        return X_train_scaled, y_train, id_train, X_test_scaled, y_test, id_test, scaler

    def _feature_scaling(X, scaler, trained_or_not):
        if trained_or_not == "not_trained":
            return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else: # "trained"
            return pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    X_cleaned = _remove_outlier_features(X)
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_cleaned), columns=X_cleaned.columns)
    X_train_scaled, y_train, id_train, X_test_scaled, y_test, id_test, scaler = _get_normed_train_test(X_imputed, y, ids, split_with_ids, testset_ratio, trained_scaler, seed)

    return {
        "X_train": X_train_scaled, 
        "X_test": X_test_scaled, 
        "y_train": y_train, 
        "y_test": y_test, 
        "id_train": id_train.tolist() if isinstance(id_train, pd.Series) else id_train, 
        "id_test": id_test.tolist() if isinstance(id_test, pd.Series) else id_test, 
        "scaler": scaler
    }

def build_pipline(params, model_name, seed, n_jobs=16):
    '''
    Return:
    - (sklearn.pipeline.Pipeline): A pipeline of feature selection and regression model.
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    '''
    # ## Feature selection:
    # feature_selector = FeatureSelector(
    #     method=params["fs_method"], 
    #     thresh_method=params["fs_thresh_method"], 
    #     threshold=params["fs_threshold"], 
    #     explained_ratio=params["fs_explained_ratio"],
    #     max_feature_num=params["fs_max_feature_num"], 
    #     seed=seed, 
    #     n_jobs=n_jobs
    # )

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
        # ("feature_selector", feature_selector), 
        ("regressor", model)
    ])

def optimize_objective(trial, X, y, # default_fs_params, 
                       model_name, seed, n_jobs=16): 
    '''
    Return:
    - (float): Average mean absolute error (MAE) score across cross validation.
    '''
    # ## Set up feature selection parameters:
    # for k, v in default_fs_params.items():
    #     trial.set_user_attr(k, v)

    # if default_fs_params["fs_method"] is None:
    #     # params = {
    #     #     "fs_method": trial.suggest_categorical("fs_method", ["LassoCV", "ElasticNetCV", "RF-Permute", "ElaNet-SHAP", "RF-SHAP", "LGBM-SHAP"]), 
    #     #     "fs_thresh_method": trial.user_attrs["fs_thresh_method"], 
    #     #     "fs_max_feature_num": trial.user_attrs["fs_max_feature_num"]
    #     # }
    #     raise NotImplementedError("Currently, unspecified feature selection method is not supported.")
    # else:
    #     params = {
    #         "fs_method": trial.user_attrs["fs_method"], 
    #         "fs_thresh_method": trial.user_attrs["fs_thresh_method"], 
    #         "fs_max_feature_num": trial.user_attrs["fs_max_feature_num"]
    #     }

    # if params["fs_thresh_method"] == "fixed_threshold":
    #     params.update({
    #         "fs_threshold": trial.suggest_float("fs_threshold", 1e-5, 0.001), 
    #         "fs_explained_ratio": trial.user_attrs["fs_explained_ratio"]
    #     })

    # elif params["fs_thresh_method"] == "explained_ratio":
    #     params.update({
    #         "fs_explained_ratio": trial.suggest_float("fs_explained_ratio", 0.9, 1), 
    #         "fs_threshold": trial.user_attrs["fs_threshold"]
    #     })

    params = {}

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

def train_and_evaluate(X, y, # fs_method, thresh_method, max_feature_num, 
                       model_names, seed, n_jobs=16):
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
        # default_fs_params = {
            # "fs_method": fs_method, 
        #     "fs_thresh_method": thresh_method, 
        #     "fs_max_feature_num": max_feature_num, 
        #     "fs_explained_ratio": None, 
        #     "fs_threshold": None
        # }
        study.optimize(
            lambda trial: optimize_objective(
                trial, X, y, # default_fs_params, 
                model_name, seed, n_jobs
            ),
            n_trials=50, show_progress_bar=True
        )
        logging.info("Parameter optimization is completed :-)")

        ## Train and evaluate the model across 5-fold CV:
        logging.info("Evaluating the model with the best hyperparameters ...")
        best_params = study.best_params
        # best_params.update({
        #     k: v for k, v in default_fs_params.items() if k not in best_params.keys()
        # })
        best_pipeline = build_pipline(
            best_params, model_name, seed, n_jobs
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        mae_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)
            if np.isnan(y_pred).any(): 
                raise ValueError("Model prediction contains NaN values.")
            mae_scores.append(mean_absolute_error(y_test, y_pred))
        
        ## Storing results:
        trained_model = best_pipeline.named_steps["regressor"]

        if hasattr(trained_model, "coef_"):
            feature_importance = pd.Series(trained_model.coef_, index=X.columns)
            feature_importance.sort_values(ascending=False, key=abs, inplace=True)
        elif hasattr(trained_model, "feature_importances_"):
            feature_importance = pd.Series(trained_model.feature_importances_, index=X.columns)
            feature_importance.sort_values(ascending=False, key=abs, inplace=True)
        else:
            feature_importance = None

        results[model_name] = {
            "trained_model": trained_model,  
            "selected_features": X.columns.tolist(), 
            # "selected_features": best_pipeline.named_steps["feature_selector"].selected_features, 
            "feature_importances": feature_importance,
            # "feature_importances": best_pipeline.named_steps["feature_selector"].feature_importances, 
            "cv_scores": mae_scores, 
            "mae_mean": np.mean(mae_scores), 
            "mae_std": np.std(mae_scores)
        }
        results[model_name].update(best_params)

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

## Main function: ---------------------------------------------------------------------

def main():
    ## Parse command line arguments (and set temporary default values):
    args = define_arguments()
    args.split_with_ids = os.path.join(os.getcwd(), "..", "outputs", "train_and_test_ids.json")
    args.include_all_mappings = True

    ## Setup config and constants objects:
    constants = Constants(args)
    config = Config(args, constants)

    os.makedirs(config.out_folder)

    ## Setup logging:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename=config.logging_outpath
    )
    logging.info(f"\nStart at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    ## Copy the current Python script to the output folder:
    shutil.copyfile(
        src=os.path.abspath(__file__), 
        dst=os.path.join(config.out_folder, os.path.basename(__file__))
    )
    logging.info("The current python script is copied to the output folder :-)")

    ## Save the description of the current execution as a JSON file:
    save_description(args, config, constants)

    if args.use_prepared_data is not None: 
        logging.info("Loading the prepared dataset ...")
        DF_prepared = pd.read_csv(args.use_prepared_data)
        
    else:
        logging.info("Loading and processing the raw dataset ...")
        data_manager = DataManager(
            raw_data_path=config.raw_data_path, 
            inclusion_data_path=config.inclusion_data_path
        )
        data_manager.load_data()
        DF_prepared = data_manager.DF

        if config.balancing_method is not None:
            logging.info(f"Making balanced datasets using '{config.balancing_method}' method ...")
            
            balancing_outdir = os.path.join(config.syndata_folder, f"{config.balancing_method.lower()}_{config.balancing_groups}_{config.balance_to_num}")
            while os.path.exists(balancing_outdir):
                balancing_outdir = balancing_outdir + "+"
            os.makedirs(balancing_outdir)

            data_manager.make_data_balanced(
                method=config.balancing_method, 
                balancing_groups=config.balancing_groups,
                age_bin_dict=constants.age_groups["cut-8-wais"], 
                balance_to_num=config.balance_to_num, 
                seed=config.seed, 
                out_dir=balancing_outdir
            )
            DF_prepared = data_manager.DF_balanced

    logging.info("Saving the prepared dataset ...")
    DF_prepared.to_csv(config.prepared_data_outpath, index=False) # copy to the output folder

    ## If user only wants to prepare the data:
    if args.prepare_data_and_exit:
        logging.info("Data preparation is completed. Exiting the program ...")
        return
    
    ## Divide the dataset into groups, define their labels, and separately perform preprocessing:
    group_name_list, sub_DF_list = divide_dataset(DF_prepared, config, args.by_gender)
    preprocessed_data_dicts = {}
    train_ids, test_ids = [], []

    for group_name, sub_DF in zip(group_name_list, sub_DF_list):
        if sub_DF.empty:
            logging.warning(f"Oh no! Data subset of the '{group_name}' group is empty :-S")
            preprocessed_data_dicts[group_name] = None

        else:
            logging.info(f"Preprocessing data subset of the {group_name} group ...")

            if args.pretrained_model_folder is not None:
                logging.info("Loading pre-trained scaler ...")
                try:
                    with open(os.path.join("outputs", args.pretrained_model_folder, f"scaler_{group_name}.pkl"), 'rb') as f:
                        trained_scaler = pickle.load(f)
                    
                except FileNotFoundError:
                    logging.warning("Pre-trained scaler not found, build a new MinMaxScaler.")
                    trained_scaler = None
            else:
                trained_scaler = None
    
            ## Preprocess the dataset and save the outputs as a dictionary:
            preprocessed_data_dicts[group_name] = preprocess_divided_dataset(
                X=sub_DF.drop(columns=["ID", "BASIC_INFO_AGE"]), 
                y=sub_DF["BASIC_INFO_AGE"], 
                ids=sub_DF["ID"], 
                split_with_ids=config.split_with_ids, 
                testset_ratio=config.testset_ratio, 
                trained_scaler=trained_scaler, 
                seed=config.seed
            ) 
            train_ids.append(preprocessed_data_dicts[group_name]["id_train"])
            test_ids.append(preprocessed_data_dicts[group_name]["id_test"])

            ## Save the scaler to the output folder:
            with open(config.scaler_outpath_format.format(group_name), 'wb') as f:
                pickle.dump(preprocessed_data_dicts[group_name]["scaler"], f)

    ## Save the IDs of participants in the training and testing sets (ungrouped):
    logging.info("Saving the IDs of participants in the training and testing sets ...")
    split_with_ids = {
        "Train": sum(train_ids, []), # ensure a flat list
        "Test": sum(test_ids, [])
    }
    with open(config.split_ids_outpath, 'w', encoding='utf-8') as f:
        json.dump(split_with_ids, f, ensure_ascii=False)

    logging.info("Data preprocess is completed, start training the models ...")    
    record_if_failed = [] # a list of strings to record the failed processing
    iter = 0 # iteration counter for skipping

    for group_name, data_dict in preprocessed_data_dicts.items():
        logging.info(f"Group: '{group_name}'")

        if data_dict is None: 
            logging.warning("Data subset of the current group is empty!!")
            logging.info("Unable to train models, skipping ...")
            record_if_failed.append(f"Entire {group_name}.")
            continue

        else: 
            for ori_name, ori_content in constants.domain_approach_mapping.items():
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

                        if not args.no_pca:
                            reducer_path = os.path.join("outputs", args.pretrained_model_folder, os.path.basename(config.reducer_outpath_format.format(group_name, ori_name)))
                            with open(reducer_path, 'rb') as f:
                                reducer = pickle.load(f)
                            X_train_transformed = reducer.transform(data_dict["X_train"])
                        else:
                            raise NotImplementedError("Using PCA is required for now.")
                            # X_train_selected = data_dict["X_train"].loc[:, selected_features]

                        saved_model = os.path.join("outputs", args.pretrained_model_folder, f"models_{group_name}_{ori_name}_{best_model_name}.pkl")
                        with open(saved_model, 'rb') as f:
                            trained_model = pickle.load(f)

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
                            raise NotImplementedError("Using previously selected features is not supported yet.")
                        
                        else: # do what supposed to be done
                            logging.info("Filtering features based on the domain and approach ...")
                            included_features = [ 
                                col for col in data_dict["X_train"].columns
                                if any( domain in col for domain in ori_content["domains"] )
                                and any( app in col for app in ori_content["approaches"] )
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
                                
                                if not args.no_pca:
                                    logging.info("Performing dimensionality reduction ...")
                                    reducer = FeatureReducer(seed=config.seed, n_iter=100)
                                    reducer.fit(X=X_train_included, rhcf=args.remove_highly_correlated_features)
                                    X_train_transformed = reducer.transform(X=X_train_included)

                                    logging.info("Saving the loading matrixs ...")
                                    pc_xlsx = config.pc_loadings_outpath_format.format(group_name, ori_name)
                                    with pd.ExcelWriter(pc_xlsx, engine="openpyxl", mode="w") as writer:
                                        for f in reducer.loadings.keys():
                                            reducer.loadings[f].to_excel(writer, sheet_name=f)
                                    
                                    rc_xlsx = config.rc_loadings_outpath_format.format(group_name, ori_name)
                                    with pd.ExcelWriter(rc_xlsx, engine="openpyxl", mode="w") as writer:
                                        for f in reducer.rotated_loadings.keys():
                                            reducer.rotated_loadings[f].to_excel(writer, sheet_name=f)

                                    logging.info("Saving the feature reducer object ...")
                                    reducer_path = config.reducer_outpath_format.format(group_name, ori_name)
                                    with open(reducer_path, 'wb') as f:
                                        pickle.dump(reducer, f)

                                    logging.info("Saving the dropped features ...")
                                    with open(config.dropped_features_outpath_format.format(group_name, ori_name), 'w') as f:
                                        f.write("\n".join(reducer.dropped_features))

                                else:
                                    raise NotImplementedError("Feature reduction is required for now.")
                                
                                logging.info("Training and evaluating models ...")
                                results = train_and_evaluate(
                                    X=X_train_transformed, 
                                    y=data_dict["y_train"], 
                                    # fs_method=config.feature_selection_method, 
                                    # thresh_method=config.fs_thresh_method, 
                                    # max_feature_num=args.max_feature_num, 
                                    model_names=config.included_models, 
                                    seed=config.seed, 
                                    n_jobs=constants.n_jobs
                                ) 
                                best_model_name = min(
                                    results, key=lambda x: results[x]["mae_mean"]
                                )
                                trained_model = results[best_model_name]["trained_model"]
                                
                                logging.info("Saving the best-performing model (.pkl) ...")
                                model_outpath = config.model_outpath_format.format(group_name, ori_name, best_model_name)
                                with open(model_outpath, 'wb') as f:
                                    pickle.dump(trained_model, f)

                                selected_features = results[best_model_name]["selected_features"]

                                logging.info("Saving the best selected features and their importances ...")
                                features_outpath = config.features_outpath_format.format(group_name, ori_name, "") #, results[best_model_name]["fs_method"])
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

                    if not args.no_pca:
                        y_pred_train = trained_model.predict(X_train_transformed)
                    else:
                        raise NotImplementedError("Using PCA is required for now.")
                        # X_train_selected = data_dict["X_train"].loc[:, selected_features]
                        # y_pred_train = trained_model.predict(X_train_selected)

                    pad_train = y_pred_train - data_dict["y_train"]

                    if config.age_correction_method == "Zhang et al. (2023)":
                        logging.info("Generating age-correction reference table ...")
                        correction_ref = generate_correction_ref(
                            age=data_dict["y_train"], 
                            pad=pad_train, 
                            age_groups=config.pad_age_groups, 
                            age_breaks=config.pad_age_breaks
                        )

                        logging.info("Applying age-correction to the training set ...")
                        corrected_y_pred_train = apply_age_correction(
                            predictions=y_pred_train, 
                            true_ages=data_dict["y_train"], 
                            correction_ref=correction_ref, 
                            age_groups=config.pad_age_groups, 
                            age_breaks=config.pad_age_breaks
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
                            "FeatureNames": list(selected_features) 
                        }

                    else:
                        logging.info("Evaluating the best model on the testing set ...")

                        if not args.no_pca:
                            X_test_transformed = reducer.transform(data_dict["X_test"])
                            y_pred_test = trained_model.predict(X_test_transformed)
                        else:
                            raise NotImplementedError("Using PCA is required for now.")
                            # X_test_selected = data_dict["X_test"].loc[:, selected_features]                       
                            # y_pred_test = trained_model.predict(X_test_selected)

                        y_pred_test = pd.Series(y_pred_test, index=data_dict["y_test"].index)
                        pad = y_pred_test - data_dict["y_test"]

                        if config.age_correction_method == "Zhang et al. (2023)":
                            logging.info("Applying age-correction to the testing set ...")
                            corrected_y_pred_test = apply_age_correction(
                                predictions=y_pred_test, 
                                true_ages=data_dict["y_test"], 
                                correction_ref=correction_ref, 
                                age_groups=config.pad_age_groups, 
                                age_breaks=config.pad_age_breaks
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
                            "FeatureNames": list(selected_features)
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

    logging.info(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
if __name__ == "__main__":
    main()
    print("\nDone!\n")