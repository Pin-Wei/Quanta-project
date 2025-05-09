#!/usr/bin/python

import os
import argparse
import logging
import json
import copy
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sdv.metadata import Metadata
from rdt.transformers.numerical import FloatFormatter
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

from age_pred_model import convert_np_types

parser = argparse.ArgumentParser(description="")
parser.add_argument("-s", "--syn_method", type=int, default=0, 
                    help="The synthesize method to be used (0: 'ctgan', 1: 'tvae').")
parser.add_argument("-bg", "--balancing_groups", type=int, default=1, 
                    help="Group segmentation approach for balancing (0: 'wais_8_seg', 1: 'cut_44-45').")
parser.add_argument("-f", "--folder", type=str, default=None, 
                    help="The folder where the output files will be stored.")
args = parser.parse_args()

## Classes: ===========================================================================

class Config:
    def __init__(self):
        self.syn_method = ["ctgan", "tvae"][args.syn_method]
        self.data_file_path = os.path.join("rawdata", "DATA_ses-01_2024-12-09.csv")
        self.inclusion_file_path = os.path.join("rawdata", "InclusionList_ses-01.csv")
        if args.folder is not None:
            self.output_dir = os.path.join("syndata", args.folder)
        else:
            self.output_dir = os.path.join("syndata", f"{self.syn_method}_{datetime.today().strftime('%Y-%m-%d')}")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # else:
        #     self.output_dir = f"{self.output_dir}_{datetime.today().strftime('%H.%M.%S')}"
        #     os.makedirs(self.output_dir)
        self.log_file_path = os.path.join(self.output_dir, "log.txt")
        self.output_metadata_format = os.path.join(self.output_dir, "metadata_{}.json")
        self.output_model_format = os.path.join(self.output_dir, "model_{}.pkl")
        self.output_diagnostic_format = os.path.join(self.output_dir, "diagnostic_report_{}.pkl")
        self.output_quality_format = os.path.join(self.output_dir, "quality_report_{}.pkl")
        self.output_dataset_path = os.path.join(self.output_dir, "balanced_dataset.csv")
        self.output_desc_path = os.path.join(self.output_dir, "description.json")

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
        ## The number of participants in each balanced group:
        self.N_per_group = {
            "wais_8_seg": {
                "CTGAN": 60, 
                "SMOTENC": 60, 
                "downsample": 15, 
                "bootstrap": 15
            }, 
            "cut_44-45": {
                "CTGAN": 60*4, 
                "SMOTENC": 60*4, 
                "downsample": 15*4, 
                "bootstrap": 15*4
            }
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

def make_balanced_dataset(DF, balancing_method, age_bin_dict, N_per_group, config):
    '''
    Create tabular synthetic data with the Synthetic Data Vault (SDV) library:
    + Metadata:
        - see: https://docs.sdv.dev/sdv/concepts/metadata/metadata-api
    + Transformers: 
        - see: https://docs.sdv.dev/sdv/single-table-data/modeling/customizations/preprocessing
        - and: https://docs.sdv.dev/rdt/usage/hypertransformer/configuration
    + Synthesizer 'CTGAN' (Conditional Tabular Generative Adversarial Network) 
        - see: https://github.com/sdv-dev/CTGAN/tree/main/ctgan
        - and: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer
    + Synthesizer 'TVAE' (Tabular Variational AutoEncoder)
        - see: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/tvaesynthesizer
    '''
    if not os.path.exists(config.output_dataset_path):

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
        
        ## Prepare for metadata:
        float_cols = list(DF.columns)
        for col in ["AGE-GROUP_SEX", "BASIC_INFO_AGE", "BASIC_INFO_SEX"]:
            float_cols.remove(col)
        
        ## Make balanced datasets:
        target_col = "AGE-GROUP_SEX"
        target_classes = list(DF[target_col].unique())
        DF_upsampled_list = []
        
        for t in target_classes:
            dataset_path = config.output_dataset_path.replace(".csv", f"_{t}.csv")

            if not os.path.exists(dataset_path):
                DF_real = DF[DF[target_col] == t]
                X_real = DF_real.drop(columns=[target_col])
                N_syn = N_per_group - len(X_real)
                print(f"Generating {N_syn} synthetic data for {t}...")

                ## Fill missing values (not including the 'AGE-GROUP' column):
                imputer = SimpleImputer(strategy="median")
                X_real_imputed = pd.DataFrame(imputer.fit_transform(X_real), columns=X_real.columns)
                
                ## Detect metadata and save:
                if not os.path.exists(config.output_metadata_format.format(t)):
                    metadata = Metadata.detect_from_dataframe(data=X_real_imputed, infer_sdtypes=False, infer_keys=None)
                    metadata.update_columns(column_names=float_cols, sdtype='numerical', computer_representation='Float')
                    metadata.update_columns_metadata(column_metadata={
                        "BASIC_INFO_SEX": {"sdtype": "categorical"},
                        "BASIC_INFO_AGE": {"sdtype": "numerical", "computer_representation": "Int64"}
                    })
                    metadata.save_to_json(config.output_metadata_format.format(t))
                    print(f"Metadata is saved to {config.output_metadata_format.format(t)}\n")
                else:
                    print(f"Loading metadata from {config.output_metadata_format.format(t)} ...")
                    metadata = Metadata.load_from_json(config.output_metadata_format.format(t))

                if not os.path.exists(config.output_model_format.format(t)):
                    ## Define model and modify transformers:
                    if balancing_method == "CTGAN":
                        synthesizer = CTGANSynthesizer(metadata, verbose=True)
                    elif balancing_method == "TVAE":
                        synthesizer = TVAESynthesizer(metadata, verbose=True)
                    else:
                        raise ValueError(f"Unknown balancing method: {balancing_method}")
                    
                    synthesizer.auto_assign_transformers(X_real_imputed)
                    for col in float_cols:
                        synthesizer.update_transformers(column_name_to_transformer={
                            col: FloatFormatter(learn_rounding_scheme=True)
                        })

                    ## Applying the transformations and train model (and save):
                    synthesizer.fit(X_real_imputed) # uses 'preprocess' and 'fit_processed_data' functions in succession.
                    synthesizer.save(config.output_model_format.format(t))
                    print(f"Model is saved to {config.output_model_format.format(t)}\n")
                else:
                    print(f"Loading model from {config.output_model_format.format(t)} ...")
                    if balancing_method == "CTGAN":
                        synthesizer = CTGANSynthesizer.load(config.output_model_format.format(t))
                    elif balancing_method == "TVAE":
                        synthesizer = TVAESynthesizer.load(config.output_model_format.format(t))
                    else:
                        raise ValueError(f"Unknown balancing method: {balancing_method}")

                ## Generate synthetic data:
                X_synthetic = synthesizer.sample(num_rows=N_syn)

                ## Run diagnostic and save:
                if os.path.exists(config.output_diagnostic_format.format(t)):
                    print("Removing old diagnostic report ...")
                    os.remove(config.output_diagnostic_format.format(t))
                diagnostic = DiagnosticReport()
                diagnostic.generate(
                    real_data=X_real_imputed,
                    synthetic_data=X_synthetic,
                    metadata=metadata.to_dict()["tables"]["table"]
                )
                diagnostic.save(filepath=config.output_diagnostic_format.format(t))
                print(f"Diagnostic report is saved to {config.output_diagnostic_format.format(t)}\n")

                ## Evaluate quality:
                if os.path.exists(config.output_quality_format.format(t)):
                    print("Removing old quality report ...")
                    os.remove(config.output_quality_format.format(t))
                quality = QualityReport()
                quality.generate(
                    real_data=X_real_imputed,
                    synthetic_data=X_synthetic,
                    metadata=metadata.to_dict()["tables"]["table"]
                )
                quality.save(filepath=config.output_quality_format.format(t))
                print(f"Quality report is saved to {config.output_quality_format.format(t)}\n")
                
                ## Concatenate real and synthetic data:
                X_synthetic.insert(1, "R_S", "Synthetic") 
                X_real_imputed.insert(1, "R_S", "Real")
                DF_upsampled = pd.concat([X_synthetic, X_real_imputed], axis=0).copy() # add .copy() to de-fragmented

                DF_upsampled.to_csv(dataset_path, index=False)
            else:
                DF_upsampled = pd.read_csv(dataset_path)

            DF_upsampled_list.append(DF_upsampled)

        DF_balanced = pd.concat(DF_upsampled_list, axis=0).reset_index(drop=True)
        DF_balanced.insert(0, "ID", [ f"sub-{x:04d}" for x in DF_balanced.index ])

        DF_balanced.to_csv(config.output_dataset_path, index=False)
        print(f"{balancing_method} dataset is saved to {config.output_dataset_path}")

    else:
        print(f"{balancing_method} dataset exists")
        print("If you want to create a new one, please specify a different folder name.")

## Main: ==============================================================================

if __name__ == "__main__":

    ## Setup config and constant objects:
    config = Config()
    constant = Constants()

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(levelname)s %(message)s', 
        filename=config.log_file_path
    )

    ## Define the sampling method and number of participants per balanced group:
    balancing_method = config.syn_method.upper()
    balancing_groups = ["wais_8_seg", "cut_44-45"][args.balancing_groups]
    age_bin_dict = constant.age_groups[balancing_groups]
    N_per_group = constant.N_per_group[balancing_groups][balancing_method]

    ## Load data and make balanced dataset:
    DF = load_and_merge_datasets(
        data_file_path=config.data_file_path, 
        inclusion_file_path=config.inclusion_file_path
    )    

    make_balanced_dataset(
        DF=copy.deepcopy(DF), 
        balancing_method=balancing_method, 
        age_bin_dict=age_bin_dict, 
        N_per_group=N_per_group, 
        config=config
    )

    desc = {
        "RawDataVersion": config.data_file_path, 
        "InclusionFileVersion": config.inclusion_file_path, 
        "DataBalancingMethod": balancing_method, 
        "BalancingGroups": balancing_groups, 
        "NumPerBalancedGroup": N_per_group
    }

    desc = convert_np_types(desc)
    with open(config.output_desc_path, 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False)

## Archived code: =====================================================================

# def build_cols_relationships(col_names):
#     relationship_dict = {
#         "STR_GM": [], "STR_WM": [], "REST_MRI": [], "REST_EEG": [], 
#         "MRI_MOT": [], "MRI_MEM": [], "MRI_LAN": [], 
#         "BEH_MOT": [], "BEH_MEM": [], "BEH_LAN": [], 
#         "ST_MOT": [], "ST_MEM": [], "ST_LAN": [], 
#         "EEG_MOT": [], "EEG_MEM": [], "EEG_LAN": []
#     }
#     for col in col_names:
#         if col.startswith("STRUCTURE_"):
#             if "GM" in col:
#                 relationship_dict["STR_GM"].append(col)
#             elif "WM" in col:
#                 relationship_dict["STR_WM"].append(col)
#         elif col.startswith("RESTING_"):
#             if "MRI" in col:
#                 relationship_dict["REST_MRI"].append(col)
#             elif "EEG" in col:
#                 relationship_dict["REST_EEG"].append(col)
#         elif col.startswith("MOTOR_"):
#             if "MRI" in col:
#                 relationship_dict["MRI_MOT"].append(col)
#             elif "BEH" in col:
#                 relationship_dict["BEH_MOT"].append(col)
#             elif "ST" in col:
#                 relationship_dict["ST_MOT"].append(col)
#             elif "EEG" in col:
#                 relationship_dict["EEG_MOT"].append(col)
#         elif col.startswith("MEMORY_"):
#             if "MRI" in col:
#                 relationship_dict["MRI_MEM"].append(col)
#             elif "BEH" in col:
#                 relationship_dict["BEH_MEM"].append(col)
#             elif "ST" in col:
#                 relationship_dict["ST_MEM"].append(col)
#             elif "EEG" in col:
#                 relationship_dict["EEG_MEM"].append(col)
#         elif col.startswith("LANGUAGE_"):
#             if "MRI" in col:
#                 relationship_dict["MRI_LAN"].append(col)
#             elif "BEH" in col:
#                 relationship_dict["BEH_LAN"].append(col)
#             elif "ST" in col:
#                 relationship_dict["ST_LAN"].append(col)
#             elif "EEG" in col:
#                 relationship_dict["EEG_LAN"].append(col)

#     return relationship_dict

