#!/usr/bin/python

import os
from sys import argv
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from itertools import product
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LassoCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import json

## Define the configuration and constants ========================================

class Config:
    data_file_path = os.path.join("rawdata", "DATA_ses-01_2024-12-09.csv")
    inclusion_file_path = os.path.join("rawdata", "InclusionList_ses-01.csv")

    age_method = ["cut_at_40", "wais_7_seg"][int(argv[1])]
    sep_sex = [True, False][int(argv[2])]    
    feature_selection_model = ["LassoCV", "RF", "XGBR"][1]
    select_n_features = 20
    testset_ratio = 0
    pad_method = ["wais_7_seg", "every_5_yrs"][0]

    out_folder = os.path.join("outputs", datetime.today().strftime('%Y-%m-%d_%H.%M.%S'))
    description_outpath = os.path.join(out_folder, "description.json")
    model_outpath_template = os.path.join(out_folder, "models_groupname_modeltype.pkl")
    results_outpath_template = os.path.join(out_folder, "results_groupname_oriname.json")
    logging_outpath = os.path.join(out_folder, "log.txt")
    failure_record_outpath = os.path.join(out_folder, "failure_record.txt")

class Constants:
    age_groups = { # The age groups defined by different methods
        "cut_at_40": {
            "le-40" : ( 0, 40),    # less than or equal to
            "ge-41" : (41, np.inf) # greater than or equal to
        }, 
        "wais_7_seg": {
            "le-24": ( 0, 24), 
            "25-29": (25, 29), 
            "30-34": (30, 34),
            "35-44": (35, 44), 
            "45-54": (45, 54), 
            "55-64": (55, 64), 
            "ge-65": (65, np.inf)
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
    domain_approach_mapping = { # The correspondence between domains and approaches
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
            "approaches": ["EEG", "MRI"]  # Include domain-specific MRI features as fMRI
        }
    }
    model_names = [ # The names of models to evaluate
        "ElasticNet", 
        "RF",   # RandomForestRegressor
        "CART", # DecisionTreeRegressor
        "LGBM", # lgb.LGBMRegressor
        "XGBM"  # xgb.XGBRegressor
    ]

## Define functions =============================================================

def load_and_preprocess_data(data_file_path, inclusion_file_path, age_boundaries, age_bin_labels, sep_sex=False, seed=42, testset_ratio=0.3):
    '''
    Return: 
    - preprocessed_grouped_datasets (dict): 
        A dictionary whose keys are defined by "sub_df_labels" 
        and values are a dictionary of train-test splited data, storing 
        the standardized feature values, age, and ID numbers of participants.
    '''
    logging.info("Loading and preprocessing data...")

    ## Load the main dataset:
    df = pd.read_csv(data_file_path)

    ## Load the file marking whether a data has been collected from individual participants:
    inclusion_df = pd.read_csv(inclusion_file_path)

    ## Only include participants with MRI data:
    inclusion_df = inclusion_df.query("MRI == 1")

    ## Ensure consistent ID column names:
    if "BASIC_INFO_ID" in df.columns:
        df = df.rename(columns={"BASIC_INFO_ID": "ID"})

    ## Merge the two dataframes to apply inclusion criteria:
    df = pd.merge(df, inclusion_df[["ID"]], on="ID", how='inner')

    ## Divide the data into multiple subsets according to the given age ranges:
    if sep_sex == False:
        sub_df_labels = age_bin_labels
        sub_df_list = [
            df[df["BASIC_INFO_AGE"].between(lower_b, upper_b)] 
            for lower_b, upper_b in age_boundaries
        ]
    else: # Additionally, separate the data by sex:
        sub_df_labels = [
            f"{age_group}_{sex}" for age_group, sex in list(product(age_bin_labels, ["M", "F"]))
        ]
        sub_df_list = [
            df[df["BASIC_INFO_AGE"].between(lower_b, upper_b) & df["BASIC_INFO_SEX"] == sex] 
            for (lower_b, upper_b), sex in list(product(age_boundaries, [1, 2]))
        ]

    ## Separately preprocess different data subsets: 
    preprocessed_grouped_datasets = {}
    for group_name, sub_df in zip(sub_df_labels, sub_df_list):

        if sub_df.empty:
            preprocessed_grouped_datasets[group_name] = None
        else:
            sub_df = sub_df.reset_index(drop=True)
            X = sub_df.drop(columns=["ID", "BASIC_INFO_AGE"])
            y = sub_df["BASIC_INFO_AGE"]
            ids = sub_df["ID"]

            ## Handling missing values
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            ## Split the data into training and testing sets and standardize the features:
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

            preprocessed_grouped_datasets[group_name] = {
                "X_train": X_train_scaled, 
                "X_test": X_test_scaled, 
                "y_train": y_train, 
                "y_test": y_test, 
                "id_train": id_train.reset_index(drop=True), 
                "id_test": id_test.reset_index(drop=True)
            }

    return preprocessed_grouped_datasets

def feature_selection(X, y, model_name="LassoCV", n_features=20, nfold=5, seed=42):
    '''
    Inputs:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series)   : Target variable (i.e., age).
    - model_name (str): A key that specify the model to use for feature selection.
    - n_features (int): Number of features to select.

    Return:
    - (list): Names of selected features.
    '''
    logging.info(f"Selecting data features using {model_name} (max={n_features})...")

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
    
    # If the number of non-zero features is greater than "n_features", take the first "n_features":
    if len(ranked_features) > n_features:
            ranked_selected_features = ranked_features[-n_features:]
    else:
        ranked_selected_features = ranked_features

    return list(ranked_selected_features)

def optimize_hyperparameters(trial, X, y, model_name, seed=42): 
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

def train_and_evaluate(X, y, model_names, seed=42, optimize_trials=50):
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
            lambda trial: optimize_hyperparameters(trial, X, y, model_name), 
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
    df = pd.DataFrame({"Age": age, "PAD": pad})

    ## Assign age labels based on the given age bin edges:
    df["Group"] = pd.cut(df["Age"], bins=age_breaks, labels=age_groups)

    ## Calculate the mean and standard deviation of PAD for each age group:
    correction_ref = df.groupby("Group", observed=True)["PAD"].agg(['mean', 'std']).reset_index()
    
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

## Main program =================================================================

def main():

## Initialiation ----------------------------------------------------------------

    config = Config()
    constant = Constants()

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename=config.logging_outpath
    ) # https://zx7978123.medium.com/python-logging-%E6%97%A5%E8%AA%8C%E7%AE%A1%E7%90%86%E6%95%99%E5%AD%B8-60be0a1a6005

    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)
    
    desc = {
        "DataVersion": config.data_file_path, 
        "InclusionVersion": config.inclusion_file_path, 
        "SexSeparated": config.sep_sex, 
        "AgeGroups": age_bin_labels, 
        "CorrectionAgeGroups": pad_age_groups, 
        "TestsetRatio": config.testset_ratio, 
        "FeatureOrientations": constant.domain_approach_mapping.keys(),
        "FeatureSelectionModel": config.feature_selection_model, 
        "OptimizedModels": constant.model_names
    }
    desc = convert_np_types(desc)

    with open(config.description_outpath, 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False)

    logging.info("The description of the current run has been saved as a JSON file.")

    age_bin_labels = list(constant.age_groups[config.age_method].keys())
    age_boundaries = list(constant.age_groups[config.age_method].values()) 
    pad_age_groups = list(constant.age_groups[config.pad_method].keys())
    pad_age_breaks = [ x for x, _ in list(constant.age_groups[config.pad_method].values()) ] + [ np.inf ]
    
    record_if_failed = []

## Load data, split into groups, and preprocess -----------------------------------

    preprocessed_grouped_datasets = load_and_preprocess_data(
        data_file_path=config.data_file_path, 
        inclusion_file_path=config.inclusion_file_path, 
        age_boundaries=age_boundaries, 
        age_bin_labels=age_bin_labels, 
        sep_sex=config.sep_sex, 
        testset_ratio=config.testset_ratio
    )

## Feature selection -------------------------------------------------------------

    for group_name, group_data in preprocessed_grouped_datasets.items():

        if group_data is None:
            logging.warning(f"Unable to process data for group '{group_name}'.")
            record_if_failed.append(f"Entire {group_name}.")
            continue
        else:
            ## Filter the features based on the domain and approach:
            for ori_name, ori_content in constant.domain_approach_mapping.items():
                logging.info(f"Processing group: {group_name}, type: {ori_name}")

                included_features = [
                    col for col in group_data["X_train"].columns
                    if any( domain in col for domain in ori_content["domains"] )
                    and any( app in col for app in ori_content["approaches"] )
                    and "RESTING" not in col
                ]

                if ori_name == "FUNCTIONAL": # exclude "STRUCTURE" features
                    included_features = [ col for col in included_features if "STRUCTURE" not in col ]
                
                if len(included_features) == 0: # check if the collection is empty
                    logging.warning(f"There are no available features for orientation '{ori_name}' in group '{group_name}'.")
                    record_if_failed.append(f"{ori_name} of {group_name}.")
                    continue
                else: 
                    if ori_name == "BEH": # if the orientation is "BEH", use all features
                        selected_features = included_features

                    else: # otherwise, select features using the specified model
                        selected_features = feature_selection(
                            X=group_data["X_train"].loc[:, included_features], 
                            y=group_data["y_train"], 
                            model_name=config.feature_selection_model, 
                            n_features=config.select_n_features
                        )

## Find the best model and save its parameters -----------------------------------

                    X_train_selected = group_data["X_train"].loc[:, selected_features]

                    if X_train_selected.empty: 
                        logging.warning(f"After feature selection, there are no available features for orientation '{ori_name}' in group '{group_name}'.")
                        record_if_failed.append(f"{ori_name} of {group_name} after feature selection.")
                        continue
                    else:                    
                        results = train_and_evaluate(
                            X=X_train_selected, 
                            y=group_data["y_train"], 
                            model_names=constant.model_names
                        ) # Including ...
                            # the mean and standard deviation of MAE scores, ...
                            # the best model, and the best hyperparameters.

                        best_model_name = min(results, key=lambda x: results[x]["mae_mean"])
                        best_model = results[best_model_name]["best_model"]

## Generate age-correction reference table ----------------------------------------

                        y_pred_train = best_model.predict(X_train_selected)
                        pad_train = y_pred_train - group_data["y_train"]
                        
                        correction_ref = generate_correction_ref(
                            age=group_data["y_train"], 
                            pad=pad_train, 
                            age_groups=pad_age_groups, 
                            age_breaks=pad_age_breaks
                        )

## Apply the model, apply age-correction, and save the results --------------------
                        
                        if group_data["X_test"].empty: # apply the model to the training set
                            corrected_y_pred_train = apply_age_correction(
                                predictions=y_pred_train, 
                                true_ages=group_data["y_train"], 
                                correction_ref=correction_ref, 
                                age_groups=pad_age_groups, 
                                age_breaks=pad_age_breaks
                            )
                            corrected_y_pred_train = pd.Series(corrected_y_pred_train, index=group_data["y_train"].index)
                            padac_train = corrected_y_pred_train - group_data["y_train"]

                            save_results = {
                                "Model": best_model_name, 
                                "MeanTrainMAE": results[best_model_name]["mae_mean"], 
                                "NumberOfSubjs": len(group_data["id_train"]), 
                                "Note": "Train and test sets are the same.", 
                                "Age": list(group_data["y_train"]), 
                                "PredictedAge": list(y_pred_train), 
                                "PredictedAgeDifference": list(pad_train), 
                                "CorrectedPAD": list(padac_train), 
                                "CorrectedPredictedAge": list(corrected_y_pred_train), 
                                "AgeCorrectionTable": correction_ref.to_dict(orient='records'), 
                                "NumberOfFeatures": len(selected_features), 
                                "FeatureNames": selected_features, 
                            }
                        else:
                            X_test_selected = group_data["X_test"].loc[:, selected_features]                       
                            y_pred_test = best_model.predict(X_test_selected)
                            y_pred_test = pd.Series(y_pred_test, index=group_data["y_test"].index)
                            pad = y_pred_test - group_data["y_test"]
                            corrected_y_pred_test = apply_age_correction(
                                predictions=y_pred_test, 
                                true_ages=group_data["y_test"], 
                                correction_ref=correction_ref, 
                                age_groups=pad_age_groups, 
                                age_breaks=pad_age_breaks
                            )
                            corrected_y_pred_test = pd.Series(corrected_y_pred_test, index=group_data["y_test"].index)
                            padac = corrected_y_pred_test - group_data["y_test"]

                            save_results = {
                                "Model": best_model_name, 
                                "MeanTrainMAE": results[best_model_name]["mae_mean"], 
                                "NumberOfTraining": len(group_data["id_train"]), 
                                "NumberOfTesting": len(group_data["id_test"]), 
                                "Age": list(group_data["y_test"]), 
                                "PredictedAge": list(y_pred_test), 
                                "PredictedAgeDifference": list(pad), 
                                "CorrectedPAD": list(padac), 
                                "CorrectedPredictedAge": list(corrected_y_pred_test), 
                                "AgeCorrectionTable": correction_ref.to_dict(orient='records'), 
                                "NumberOfFeatures": len(selected_features), 
                                "FeatureNames": selected_features, 
                            }

                        save_results = convert_np_types(save_results)
                        fp1 = config.results_outpath_template.replace("groupname", group_name).replace("oriname", ori_name)
                        with open(fp1, 'w', encoding='utf-8') as f:
                            json.dump(save_results, f, ensure_ascii=False)

                        logging.info(f"Model prediction have been saved as JSON files for group '{group_name}' and orientation '{ori_name}'.")

                        fp2 = config.model_outpath_template.replace("groupname", group_name).replace("modeltype", best_model_name)
                        with open(fp2, 'wb') as f:
                            pickle.dump(best_model, f)

                        logging.info(f"The trained model have been saved for group '{group_name}' and orientation '{ori_name}'.")
    
    with open(config.failure_record_outpath, 'w') as f:
        f.write("\n".join(record_if_failed))
    
    logging.info("The record of failed processing have been saved.")

if __name__ == "__main__":
    main()


