#!/usr/bin/python

# python age_pred_model.py [-age] [-sex] [-u] [-d] [-b] [-n] [-tsr] [-iam] [-oam] [-fsm] [-epr] [-pmf] [-i] [-s]

import os
import numpy as np
import pandas as pd
import copy
from imblearn.over_sampling import SMOTENC 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer

## Classes: ===========================================================================

class Config:
    def __init__(self):
        self.data_file_path = os.path.join("rawdata", "DATA_ses-01_2024-12-09.csv")
        self.inclusion_file_path = os.path.join("rawdata", "InclusionList_ses-01.csv")

class Constants:
    def __init__(self):
        ## The age groups defined by different methods:
        self.age_groups = { 
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
            "SMOTENC": 60, 
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
    # N_per_group = 0 if N_per_group is None else N_per_group
    for t in target_classes:
        sub_DF = DF[DF[target_col] == t]
        sub_X = sub_DF.drop(columns=[target_col])
        imputer = SimpleImputer(strategy="median")
        sub_DF_imputed = pd.DataFrame(imputer.fit_transform(sub_X), columns=sub_X.columns)
        sub_DF_imputed[target_col] = sub_DF[target_col].reset_index(drop=True)
        DF_imputed_list.append(sub_DF_imputed)
        # if (balancing_method == "SMOTENC") and (len(sub_DF_imputed) > N_per_group): 
        #     N_per_group = len(sub_DF_imputed)
        # elif (balancing_method == "downsample" | balancing_method == "bootstrap") and (len(sub_DF_imputed) < N_per_group):
        #     N_per_group = len(sub_DF_imputed)
    DF_imputed = pd.concat(DF_imputed_list)
    DF_imputed.reset_index(drop=True, inplace=True)

    ## Make balanced datasets:    
    if balancing_method == "SMOTENC":
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

## Main: ==============================================================================

if __name__ == "__main__":

    ## Setup config and constant objects:
    config = Config()
    constant = Constants()

    ## Define the sampling method and number of participants per balanced group:
    balancing_method = "SMOTENC"
    N_per_group = constant.N_per_group[balancing_method]

    ## Load data and make balanced dataset:
    DF = load_and_merge_datasets(
        data_file_path=config.data_file_path, 
        inclusion_file_path=config.inclusion_file_path
    )

    for seed in range(1, 10000): 
        target_col, DF_prepared = make_balanced_dataset(
            DF=copy.deepcopy(DF), 
            balancing_method=balancing_method, 
            age_bin_dict=constant.age_groups["wais_8_seg"], 
            N_per_group=N_per_group, 
            seed=seed
        )
        DF_prepared.drop(columns=[target_col], inplace=True)

        mark_synthetic_data(
            DF, DF_prepared
        ).to_csv(
            os.path.join("syndata", f"smotenc_balanced_dataset_{seed}.csv"),
            index=False
        )



