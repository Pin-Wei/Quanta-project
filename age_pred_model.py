#!/usr/bin/python

# Usage: python age_pred_model.py -age [0|2] -iam (-pmf <pretrained_model_folder>)

import os
import re
import sys
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
# import shap # SHapley Additive exPlanations
import optuna
import optunahub
# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform

from imblearn.over_sampling import SMOTENC 
from imblearn.under_sampling import RandomUnderSampler
from factor_analyzer import Rotator

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
# from sklearn.feature_selection import RFECV, SelectFromModel
# from sklearn.inspection import permutation_importance
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
## Data preparation:
parser.add_argument("-dat", "--prepare_data_and_exit", action="store_true", default=False, 
                    help="Only make the data, do not train any model.")
parser.add_argument("-upd", "--use_prepared_data", type=str, default=None, 
                    help="File path of the data to be used (.csv).")
## Balancing data such that all groups have the same number of participants:
parser.add_argument("-up", "--smotenc", action="store_true", default=False, 
                    help="Up-sample the data using SMOTENC.")
parser.add_argument("-down", "--downsample", action="store_true", default=False, 
                    help="Down-sample the data without replacement.")
parser.add_argument("-boot", "--bootstrap", action="store_true", default=False, 
                    help="Down-sample the data with replacement (i.e., bootstrapping).")
parser.add_argument("-bg", "--balancing_groups", type=int, default=0, 
                    help="The groups to be balanced.")
parser.add_argument("-n", "--sample_size", type=int, default=None, 
                    help="The number of participants to up- or down-sample to. (if None, use the default number of participants per group set in constants.n_per_balanced_g).")
## How to separate data into groups:
parser.add_argument("-age", "--age_method", type=int, default=2, 
                    help="The method to define age groups (0: 'no_cut', 1: 'cut_at_40', 2: 'cut_44-45', 3: 'wais_8_seg').")
parser.add_argument("-sex", "--by_gender", type=int, default=0, 
                    help="Whether to separate the data by gender (0: False, 1: True).")
## Split data into training and testing sets:
parser.add_argument("-tsr", "--testset_ratio", type=float, default=0.3, 
                    help="The ratio of the testing set.")
parser.add_argument("-sid", "--split_with_ids", type=str, default=None, 
                    # default=os.path.join("outputs", "train_and_test_ids.json"), 
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

## Classes: ===========================================================================

class Config:
    def __init__(self, args):
        self.source_path = os.path.dirname(os.path.abspath(__file__))
        self.data_file_path = os.path.join(self.source_path, "rawdata", "DATA_ses-01_2024-12-09.csv")
        self.inclusion_file_path = os.path.join(self.source_path, "rawdata", "InclusionList_ses-01.csv")
        self.balancing_groups = ["wais_8_seg", "cut_44-45"][args.balancing_groups]
        self.age_method = ["no_cut", "cut_at_40", "cut_44-45", "wais_8_seg"][args.age_method]
        self.by_gender = [False, True][args.by_gender]
        # self.feature_selection_method = [None, "LassoCV", "ElasticNetCV", "RF-Permute", "ElaNet-SHAP", "RF-SHAP", "LGBM-SHAP"][args.feature_selection_method]
        # self.fs_thresh_method = ["fixed_threshold", "explained_ratio"][args.fs_thresh_method]
        self.age_correction_method = ["Zhang et al. (2023)", "Beheshti et al. (2019)"][args.age_correction_method]
        self.age_correction_groups = ["wais_8_seg", "every_5_yrs"][0]
        self.seed = args.seed if args.seed is not None else np.random.randint(0, 10000)

        if args.split_with_ids is not None:
            with open(args.split_with_ids, 'r') as f:
                self.split_with_ids = json.load(f)

            self.testset_ratio = (
                len(self.split_with_ids["Test"]) / (len(self.split_with_ids["Train"]) + len(self.split_with_ids["Test"]))
            )
        else:
            self.split_with_ids = None
            self.testset_ratio = args.testset_ratio

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

        # if self.feature_selection_method is not None:
        #     folder_prefix += f"_{self.feature_selection_method}"
        # else:
        #     folder_prefix += "_Auto"

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

        self.logging_outpath = os.path.join(self.out_folder, "log.txt")
        self.description_outpath = os.path.join(self.out_folder, "description.json")
        self.prepared_data_outpath = os.path.join(self.out_folder, "prepared_data.csv")
        self.split_ids_outpath = os.path.join(self.out_folder, "train_and_test_ids.json")
        self.scaler_outpath_format = os.path.join(self.out_folder, "scaler_{}.pkl")
        self.reducer_outpath_format = os.path.join(self.out_folder, "reducer_{}_{}.pkl")
        self.pc_loadings_outpath_format = os.path.join(self.out_folder, "PC_loadings_{}_{}.xlsx")
        self.rc_loadings_outpath_format = os.path.join(self.out_folder, "RC_loadings_{}_{}.xlsx")
        self.dropped_features_outpath_format = os.path.join(self.out_folder, "dropped_features_{}_{}.txt")
        self.model_outpath_format = os.path.join(self.out_folder, "models_{}_{}_{}.pkl")
        self.features_outpath_format = os.path.join(self.out_folder, "features_{}_{}.csv")
        self.training_results_outpath_format = os.path.join(self.out_folder, "training_results_{}_{}.json")
        self.results_outpath_format = os.path.join(self.out_folder, "results_{}_{}.json")
        self.failure_record_outpath = os.path.join(self.out_folder, "failure_record.txt")

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

class FeatureReducer:
    def __init__(self, seed, n_iter=100):
        self.seed = seed
        self.n_iter = n_iter
        self.dropped_features = []
        self.n_retained_components = {}
        self.fitted_pca = {}
        self.loadings = {}
        self.rotated_loadings = {}
        self.rotated_components = {}

    def fit(self, X, rhcf=False):
        '''
        Maths underlying Principal Component Analysis (PCA):
        # https://en.wikipedia.org/wiki/Principal_component_analysis#Details
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_pca.py#L113
        
        Varimax rotation:
        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_varimax_fa.html

        <Return attributes>
        - categorized_features: a dictionary storing the features for each category.
        - n_retained_components: a dictionary storing the number of retained components for each category.
        - fitted_pca: a dictionary storing the fitted PCA models for each category.
        '''
        ## Categorize features (so that n/p >= 5)
        categorized_features = self._categorize_features(X.columns)
        self.categorized_features = categorized_features

        ## For each category
        for f, feature_list in self.categorized_features.items():
            X_splited = X.loc[:, feature_list]

            ## Remove highly correlated features
            if rhcf:
                X_splited, dropped_features = self._remove_highly_correlated_features(X_splited)
                self.dropped_features.extend(dropped_features)

            ## Perform parallel analysis to determine the number of components to retain
            n_retained = self._parallel_analysis(X_splited)
            self.n_retained_components[f] = n_retained

            ## Fit a PCA model
            pca = PCA(n_components=n_retained, random_state=self.seed)
            pca.fit(X_splited)
            self.fitted_pca[f] = pca

            ## Calculate loadings
            eigenvectors = pca.components_.T
            eigenvalues = pca.explained_variance_
            loadings = eigenvectors * np.sqrt(eigenvalues)
            self.loadings[f] = pd.DataFrame(
                data=loadings, 
                columns=[f"PC{i+1}" for i in range(n_retained)], 
                index=list(X_splited.columns)
            )

            if loadings.shape[1] > 1: # Apply varimax rotation
                rotator = Rotator(method="varimax")
                rotated_loadings = rotator.fit_transform(np.array(loadings))
                self.rotated_loadings[f] = pd.DataFrame(
                    data=rotated_loadings, 
                    columns=[f"RC{i+1}" for i in range(n_retained)], 
                    index=list(X_splited.columns)
                )
                self.rotated_components[f] = rotated_loadings @ np.diag(1 / np.sqrt(eigenvalues))
                
        return self

    def transform(self, X):
        '''
        <Return> a dataframe containing the transformed features.
        '''
        component_names, component_values = [], []

        ## For each category, transform the input data using the fitted reducer model
        for f, feature_list in self.categorized_features.items():
            selected_features = [ f for f in feature_list if f not in self.dropped_features ]
            X_splited = X.loc[:, selected_features].to_numpy()
            pca = self.fitted_pca[f]
            X_centered = X_splited - pca.mean_

            if f in self.rotated_components.keys():
                X_transformed = X_centered @ self.rotated_components[f]
                for i in range(X_transformed.shape[1]):
                    component_names.append(f"{f}_RC{i+1}")
                    component_values.append(X_transformed[:, i])
            else:
                X_transformed = X_centered @ pca.components_.T
                for i in range(X_transformed.shape[1]):
                    component_names.append(f"{f}_PC{i+1}")
                    component_values.append(X_transformed[:, i])

        return pd.DataFrame(
            data=np.column_stack(component_values), 
            columns=component_names, 
            index=X.index
        )
    
    def _categorize_features(self, included_features):
        '''
        Knowledge-based categorization of features.
        '''
        categorized_features = {}
        lower_categorized_GM = { k.lower(): v for k, v in self._categorized_GM_regions().items() }
        
        for feature in included_features:
            if feature.startswith("STRUCTURE"):
                category, hemi, area, measure = feature.split("_")[3::]
                if category == "NULL":
                    f = f"STR_{category}_{measure}"
                else:
                    if (category == "GM") and (measure != "FA"): 
                        region = lower_categorized_GM[area.lower()]

                        if region == "I-C":
                            f = f"STR_{category}_{region}_{measure}"
                        else:
                            f = f"STR_{category}_{hemi[0]}_{region}_{measure}"
                    else:
                        f = f"STR_{category}_{hemi[0]}_{measure}"
            else:
                domain, task, measure, condition = feature.split("_")[:4]
                measure = "fMRI" if measure == "MRI" else measure                

                if domain == "LANGUAGE":
                    f = f"{measure}_{domain.lower()}"
                    
                elif (measure == "EEG") and (task == "OSPAN") and ("Diff" in condition):
                    f = f"{measure}_{domain.lower()}_{task.lower()}-diff"

                elif (measure == "EEG") and (task == "GOFITTS"):
                    suffix = re.sub(r"[0-9]+", "", condition).replace("ID", "").replace("W", "").replace("Slope", "-slope")
                    f = f"{measure}_{domain.lower()}_{task.lower()} {suffix.lower()}"
                
                else:
                    f = f"{measure}_{domain.lower()}_{task.lower()}"

            if f in categorized_features.keys():
                categorized_features[f].append(feature)
            else:
                categorized_features.update({f: [feature]})
        
        return categorized_features

    def _categorized_GM_regions(self):
        return {
            "GySulFrontoMargin": "F-T", # Fronto-marginal gyrus (of Wernicke) and sulcus
            "GySulSubCentral": "F-T", # Subcentral gyrus (central operculum) and sulci
            "GySulTransvFrontopol": "F-T", # Transverse frontopolar gyri and sulci
            "GyFrontInfOpercular": "F-T", # Opercular part of the inferior frontal gyrus
            "GyFrontInfObital": "F-T", # Orbital part of the inferior frontal gyrus
            "GyFrontInfTriangul": "F-T", # Triangular part of the inferior frontal gyrus
            "GyFrontMiddle": "F-T", # Middle frontal gyrus (F2)
            "GyFrontSup": "F-T", # Superior frontal gyrus (F1)
            "GyOrbital": "F-T", # Orbital gyri
            "GyRectus": "F-T", # Straight gyrus, Gyrus rectus
            "LateralFisAnterorHorizont": "F-T", # Horizontal ramus of the anterior segment of the lateral sulcus (or fissure)
            "LateralFisAnterorVertical": "F-T", # Vertical ramus of the anterior segment of the lateral sulcus (or fissure)
            "LateralFisPost": "F-T", # Posterior ramus (or segment) of the lateral sulcus (or fissure)
            "SulFrontInferior": "F-T", # Inferior frontal sulcus
            "SulFrontMiddle": "F-T", # Middle frontal sulcus
            "SulFrontSuperior": "F-T", # Superior frontal sulcus
            "SulOrbitalLateral": "F-T", # Lateral orbital sulcus
            "SulOrbitalMedialOlfact": "F-T", # Medial orbital sulcus (olfactory sulcus)
            "SulOrbitalHshaped": "F-T", # Orbital sulci (H-shaped sulci)
            "SulSubOrbital": "F-T", # Suborbital sulcus (sulcus rostrales, supraorbital sulcus)
            "GyOccipitalTemporalLateralFusifor": "F-T", # Lateral occipito-temporal gyrus (fusiform gyrus, O4-T4)
            "GyOccipitalTemporalMedialParahip": "F-T", # Parahippocampal gyrus, parahippocampal part of the medial occipito-temporal gyrus, (T5)
            "GyTemporalSuperiorGyTemporalTransv": "F-T", # Anterior transverse temporal gyrus (of Heschl)
            "GyTemporalSuperiorLateral": "F-T", # Lateral aspect of the superior temporal gyrus
            "GyTemporalSuperiorPlanPolar": "F-T", # Planum polare of the superior temporal gyrus
            "GyTemporalSuperiorPlanTempo": "F-T", # Planum temporale or temporal plane of the superior temporal gyrus
            "GyTemporalInferior": "F-T", # Inferior temporal gyrus (T3)
            "GyTemporalMiddle": "F-T", # Middle temporal gyrus (T2)
            "PoleTemporal": "F-T", # Temporal pole
            "SulOccipitalTemporalLateral": "F-T", # Lateral occipito-temporal sulcus
            "SulOccipitalTemporalMedialAndLingual": "F-T", # Medial occipito-temporal sulcus (collateral sulcus) and lingual sulcus
            "SulTemporalInferior": "F-T", # Inferior temporal sulcus
            "SulTemporalSuperior": "F-T", # Superior temporal sulcus (parallel sulcus)
            "SulTemporalTransverse": "F-T", # Transverse temporal sulcus
            "GySulCingulAnt": "I-C", # Anterior part of the cingulate gyrus and sulcus (ACC)
            "GySulCingulMidAnt": "I-C", # Middle-anterior part of the cingulate gyrus and sulcus (aMCC)
            "GySulCingulMidPost": "I-C", # Middle-posterior part of the cingulate gyrus and sulcus (pMCC)
            "GyCingulPostDorsal": "I-C", # Posterior-dorsal part of the cingulate gyrus (dPCC)
            "GyCingulPostVentral": "I-C", # Posterior-ventral part of the cingulate gyrus (vPCC, isthmus of the cingulate gyrus)
            "GyInsularLongSulCentralInsular": "I-C", # Long insular gyrus and central sulcus of the insula
            "GyInsularShort": "I-C", # Short insular gyri
            "GySubcallosal": "I-C", # Subcallosal area, subcallosal gyrus
            "SulCircularInsulaAnteror": "I-C", # Anterior segment of the circular sulcus of the insula
            "SulCircularInsulaInferior": "I-C", # Inferior segment of the circular sulcus of the insula
            "SulCircularInsulaSuperoir": "I-C", # Superior segment of the circular sulcus of the insula
            "SulPericallosal": "I-C", # Pericallosal sulcus (S of corpus callosum)
            "GySulOccipitalInf": "P-O", # Inferior occipital gyrus (O3) and sulcus
            "GyCuneus": "P-O", # Cuneus (O6)
            "GyOccipitalMiddle": "P-O", # Middle occipital gyrus (O2, lateral occipital gyrus)
            "GyOccipitalSup": "P-O", # Superior occipital gyrus (O1)
            "GyOccipitalTemporalMedialLingual": "P-O", # Lingual gyrus, ligual part of the medial occipito-temporal gyrus, (O5)
            "PoleOccipital": "P-O", # Occipital pole
            "SulCalcarine": "P-O", # Calcarine sulcus
            "SulCollatTransvAnterior": "P-O", # Anterior transverse collateral sulcus
            "SulCollatTransvPosterior": "P-O", # Posterior transverse collateral sulcus
            "SulOccipitalMiddleAndLunatus": "P-O", # Middle occipital sulcus and lunatus sulcus
            "SulOccipitalSuperiorAndTransversal": "P-O", # Superior occipital sulcus and transverse occipital sulcus
            "SulOccipitalAnterior": "P-O", # Anterior occipital sulcus and preoccipital notch (temporo-occipital incisure)
            "SulParietoOccipital": "P-O", # Parieto-occipital sulcus (or fissure)
            "GySulParaCentral": "P-O", # Paracentral lobule and sulcus
            "GyParietalInfAngular": "P-O", # Angular gyrus
            "GyParietalInfSupramar": "P-O", # Supramarginal gyrus
            "GyParietalSuperior": "P-O", # Superior parietal lobule
            "GyPostCentral": "P-O", # Postcentral gyrus
            "GyPreCentral": "P-O", # Precentral gyrus
            "GyPreCuneus": "P-O", # Precuneus (medial part of P1)
            "SulCentral": "P-O", # Central sulcus (Rolando's fissure)
            "SulCingulMarginalis": "P-O", # Marginal branch (or part) of the cingulate sulcus
            "SulIntermPrimJensen": "P-O", # Sulcus intermedius primus (of Jensen)
            "SulIntraParietAndParietalTrans": "P-O", # Intraparietal sulcus (interparietal sulcus) and transverse parietal sulci
            "SulPostCentral": "P-O", # Postcentral sulcus
            "SulPreCentralInferiorPart": "P-O", # Inferior part of the precentral sulcus
            "SulPreCentralSuperiorPart": "P-O", # Superior part of the precentral sulcus
            "SulSubParietal": "P-O" # Subparietal sulcus
        }
    
    def _remove_highly_correlated_features(self, X, threshold=0.9):
        corr_matrix = X.corr().abs()
        upper_cormat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [ column for column in upper_cormat.columns if any(upper_cormat[column] > threshold) ]
        return X.drop(columns=to_drop), to_drop

    def _parallel_analysis(self, X):
        '''
        # see: https://stackoverflow.com/questions/62303782/is-there-a-way-to-conduct-a-parallel-analysis-in-python
        # and: https://www.statstodo.com/ParallelAnalysis.php
        '''
        n_features = X.shape[1]
        pca = PCA(n_components=n_features, random_state=self.seed)
        eigv_raw = pca.fit(X).explained_variance_ratio_

        eigv_rand = np.zeros((self.n_iter, n_features))
        for i in range(self.n_iter):
            X_r = np.random.normal(loc=0, scale=1, size=X.shape)
            pca_rand = PCA(n_components=n_features, random_state=self.seed)
            eigv_rand[i, :] = pca_rand.fit(X_r).explained_variance_ratio_

        rand_eigv_mean = eigv_rand.mean(axis=0)
        rand_eigv_std = eigv_rand.std(axis=0)
        thresholds = rand_eigv_mean + rand_eigv_std * 1.64 # 95% confidence 
        n_retained = max(np.argwhere(eigv_raw > thresholds)) + 1

        return n_retained[0]

# class FeatureSelector:
#     def __init__(self, method, thresh_method, threshold, explained_ratio, max_feature_num, seed, n_jobs=16):
#         self.method = method
#         self.seed = seed
#         self.thresh_method = thresh_method
#         self.threshold = threshold
#         self.explained_ratio = explained_ratio
#         self.max_feature_num = max_feature_num
#         self.min_feature_num = 1 # 10
#         self.n_jobs = n_jobs

#     def fit(self, X: pd.DataFrame, y):
#         '''
#         Select features based on their importance weights.
#         - see: https://scikit-learn.org/stable/modules/feature_selection.html
#         - also: https://hyades910739.medium.com/%E6%B7%BA%E8%AB%87-tree-model-%E7%9A%84-feature-importance-3de73420e3f2

#         Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
#          + handling multicollinearity: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
#          - may not be suitable

#         SHAP (SHapley Additive exPlanations): https://shap.readthedocs.io/en/latest/
#         - see 1: https://christophm.github.io/interpretable-ml-book/shapley.html
#         - see 2: https://ithelp.ithome.com.tw/articles/10329606
#         - see 3: https://medium.com/analytics-vidhya/shap-part-1-an-introduction-to-shap-58aa087a460c
#         - see 4: https://medium.com/@msvs.akhilsharma/unlocking-the-power-of-shap-analysis-a-comprehensive-guide-to-feature-selection-f05d33698f77
#         '''
#         ## Check input:
#         assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
#         assert isinstance(y, pd.Series) or isinstance(y, np.ndarray), "y should be a pandas Series or a numpy array."
#         assert X.shape[0] == y.shape[0], "X and y should have the same number of rows."
#         assert (not np.any(np.isnan(X))) and (not np.any(np.isnan(y))), "X and y should not contain NaN values."
#         assert (not np.any(np.isinf(X))) and (not np.any(np.isinf(y))), "X and y should not contain infinite values."

#         ## Estimate feature importance:
#         if self.method == "LassoCV":
#             importances = LassoCV(
#                 cv=5, random_state=self.seed
#             ).fit(X, y).coef_

#         elif self.method == "ElasticNetCV":
#             importances = ElasticNetCV(
#                 l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 
#                 cv=5, random_state=self.seed, n_jobs=self.n_jobs
#             ).fit(X, y).coef_

#         elif self.method == "RF-Permute": # permutation importance 
#             # dist_matrix = 1 - X.corr(method="spearman").abs()
#             # dist_linkage = hierarchy.ward( # compute Ward’s linkage on a condensed distance matrix.
#             #     squareform(dist_matrix)
#             # ) 
#             # cluster_ids = hierarchy.fcluster( # form flat clusters from the hierarchical clustering defined by the given linkage matrix
#             #     Z=dist_linkage, t=2, criterion="distance" # t: distance threshold, manually selected
#             # )
#             # cid_to_fids = defaultdict(list) # cluster id to feature ids
#             # for idx, cluster_id in enumerate(cluster_ids):
#             #     cid_to_fids[cluster_id].append(idx)
#             # first_features = [ v[0] for v in cid_to_fids.values() ] # select the first feature in each cluster
#             X_train, X_test, y_train, y_test = train_test_split(
#                 # X.iloc[:, first_features], y, test_size=.3, random_state=self.seed
#                 X, y, test_size=.2, random_state=self.seed
#             )
#             rf_trained = RandomForestRegressor(
#                 random_state=self.seed, n_jobs=self.n_jobs
#             ).fit(X_train, y_train)
#             importances = permutation_importance(
#                 estimator=rf_trained, X=X_test, y=y_test, 
#                 n_repeats=10, random_state=self.seed, n_jobs=self.n_jobs
#             ).importances_mean

#         elif self.method == "LightGBM": # impurity-based feature importance
#             # importances = LGBMRegressor(
#             #     importance_type=["split", "gain"][1], random_state=self.seed
#             # ).fit(X, y).feature_importances_
#             raise NotImplementedError("LightGBM impurity-based feature importance should not be used.")

#         elif self.method.endswith("SHAP"): 

#             if self.method.startswith("ElaNet"): # ElasticNet-SHAP
#                 model = ElasticNetCV(
#                     l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 
#                     cv=5, random_state=self.seed, n_jobs=self.n_jobs
#                 ).fit(X, y)
#                 explainer = shap.Explainer(
#                     model=model.predict, 
#                     masker=X, # pass a background data matrix instead of a function
#                     algorithm="linear"
#                 )
#                 shap_values = explainer(X)

#             else:
#                 X_train, X_test, y_train, y_test = train_test_split(
#                     X, y, test_size=.2, random_state=self.seed
#                 )
#                 if self.method.startswith("RF"): # RF-SHAP
#                     shap_values = shap.TreeExplainer(
#                         model=RandomForestRegressor(
#                             random_state=self.seed, n_jobs=self.n_jobs
#                         ).fit(X_train, y_train), 
#                     ).shap_values(X_test)

#                 elif self.method.startswith("LGBM"): # LGBM-SHAP
#                     shap_values = shap.TreeExplainer(
#                         model=LGBMRegressor(
#                             max_depth=3, min_child_samples=5, 
#                             random_state=self.seed, n_jobs=self.n_jobs
#                         ).fit(X_train, y_train)
#                     ).shap_values(X_test)

#             importances = np.abs(shap_values).mean(axis=0)

#         ## Fallback:
#         if np.isnan(importances).any() or np.all(importances == 0):
#             raise ValueError("SHAP values are invalid (NaN or all zero).")

#         ## Selection:
#         feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        
#         if self.thresh_method == "explained_ratio":
#             normed_feature_imp = feature_importances / feature_importances.abs().sum()
#             cumulative_importances = normed_feature_imp.abs().cumsum()
#             num_features = np.argmax(cumulative_importances > self.explained_ratio) + 1
#             selected_feature_imp = feature_importances.head(num_features)

#         elif self.thresh_method == "fixed_threshold":
#             selected_feature_imp = feature_importances[feature_importances.abs() > self.threshold]

#         if self.max_feature_num is not None:
#             selected_feature_imp = selected_feature_imp.head(self.max_feature_num)
        
#         ## Fallback:
#         if len(selected_feature_imp) < self.min_feature_num:
#             raise ValueError("Too few features are selected. Threshold may be too strict.")

#         ## Record:
#         self.feature_importances = selected_feature_imp
#         self.selected_features = list(selected_feature_imp.index)

#         return self
    
#     def transform(self, X):
#         return X[self.selected_features]

## Functions: =========================================================================

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
    
def save_description(args, config, constants, age_bin_labels, pad_age_groups):
    '''
    Save the description of the current execution as a JSON file.
    '''
    if args.use_prepared_data: # should be balanced
        folder, _ = os.path.split(args.use_prepared_data)
        balancing_method = "--"
        balancing_groups = "--"
        n_per_balanced_g = "--"
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
                n_per_balanced_g = constants.n_per_balanced_g[balancing_groups][balancing_method]
            else:
                n_per_balanced_g = args.sample_size

    desc = {
        "Seed": config.seed, 
        "UsePreparedData": args.use_prepared_data, 
        "RawDataVersion": config.data_file_path if args.use_prepared_data is None else "--", 
        "InclusionFileVersion": config.inclusion_file_path if args.use_prepared_data is None else "--", 
        "DataBalancingMethod": balancing_method, 
        "BalancingGroups": balancing_groups, 
        "NumPerBalancedGroup": n_per_balanced_g
    }

    if not args.prepare_data_and_exit:

        ## Define the type of the model to be used for training:
        if args.training_model is not None:
            included_models = [constants.model_names[args.training_model]]
        elif args.pretrained_model_folder is not None:
            included_models = "Depend on the previous results"
        else:
            included_models = constants.model_names

        desc.update({
            "SexSeparated": config.by_gender, 
            "AgeGroups": age_bin_labels, 
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
            "IncludedOptimizationModels": included_models, 
            "SkippedIterationNum": args.ignore,  
            "AgeCorrectionMethod": config.age_correction_method, 
            "AgeCorrectionGroups": pad_age_groups
        })

    ## Save the description to a JSON file:
    desc = convert_np_types(desc)
    with open(config.description_outpath, 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False)

    logging.info("The description of the current execution is saved :-)")

    return desc
    
def load_and_modify_data(data_file_path, inclusion_file_path):
    '''
    Read the data and inclusion table, merge them to apply inclusion criteria, 
    and perform some modifications.
    '''
    DF = pd.read_csv(data_file_path)
    DF.rename(columns={"BASIC_INFO_ID": "ID"}, inplace=True) # ensure consistent ID column name
    
    inclusion_df = pd.read_csv(inclusion_file_path)
    inclusion_df = inclusion_df.query("MRI == 1") # only include participants with MRI data

    DF = pd.merge(DF, inclusion_df[["ID"]], on="ID", how='inner') # apply inclusion criteria
    
    ## Drop columns based on knowledge: 
    DF.drop(columns=[
        col for col in DF.columns 
        if ( "RESTING" in col )
        or ( col.startswith("LANGUAGE_SPEECHCOMP_BEH") and col.endswith("RT") )
        or ( col.startswith("MOTOR_GOFITTS_EEG") and (("Diff" in col) or ("Slope" in col)) )
        or ( col.startswith("MEMORY_EXCLUSION_BEH") and any( kw in col for kw in ["TarMiss", "NewFA", "NonTarFA_PROPORTION", "C2NonTarFA_RT", "C3NonTarFA_RT", "C1NewCR_PROPORTION", "C1NewCR_RTvar"] ) )
        or ( col.startswith("MEMORY_EXCLUSION_EEG") and any( kw in col for kw in ["TarHitNewCRdiff", "NonTarCRNewCRdiff"] ) )
        or ( col.startswith("MEMORY_OSPAN_EEG") and any( kw in col for kw in ["150To350", "AMPLITUDE"] ) )
        or ( col.startswith("MEMORY_MST_MRI") and any( kw in col for kw in ["OldCorSimCorDiff", "OldCorNewCorDiff", "SimCorNewCorDiff", "MD"] ) )
    ], inplace=True)

    DF.drop(columns=[
        "MEMORY_OSPAN_EEG_MathItem01_PZ_250To450_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem23_PZ_250To450_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem456_PZ_250To450_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem01_PZ_400To600_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem23_PZ_400To600_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem456_PZ_400To600_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem01_O1_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem23_O1_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem456_O1_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem01_O2_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem23_O2_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem456_O2_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem01_OZ_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem23_OZ_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem456_OZ_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem01_PZ_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem23_PZ_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem456_PZ_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem2301Diff_O1_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem2301Diff_O2_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem2301Diff_OZ_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem45601Diff_O1_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem45601Diff_O2_600To800_4To8_POWER", 
        "MEMORY_OSPAN_EEG_MathItem01_PZ_1200To1400_20To25_POWER", 
        "MEMORY_OSPAN_EEG_MathItem23_PZ_1200To1400_20To25_POWER", 
        "MEMORY_OSPAN_EEG_MathItem456_PZ_1200To1400_20To25_POWER"
    ], inplace=True)

    ## Calculate mean of HighForce and LowForce columns:
    for col in DF.columns:
        if re.match(r"MOTOR_GFORCE_MRI_.*?HighForce_.*?H_.*?_[a-zA-Z]+", col):
            pair = col.replace("High", "Low")
            DF.insert(
                DF.columns.get_loc(col), 
                col.replace("High", "Mean"), 
                DF[[col, pair]].mean(axis=1)
            )
            DF.drop(
                columns=[col, pair], 
                inplace=True
            )

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

def preprocess_grouped_dataset(X, y, ids, split_with_ids, testset_ratio, trained_scaler, seed):
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
    ## Remove features (data columns) with too many missing observations:
    n_subjs = len(X)
    na_rates = pd.Series(X.isnull().sum() / n_subjs)
    Q1 = na_rates.quantile(.25)
    Q3 = na_rates.quantile(.75)
    IQR = Q3 - Q1
    outliers = na_rates[na_rates > (Q3 + IQR * 1.5)]
    X_cleaned = X.drop(columns=outliers.index)

    ## Fill missing values:
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_cleaned), columns=X_cleaned.columns)
    
    ## Split into training and testing sets, and then apply feature scaling:
    if trained_scaler is None:
        scaler = StandardScaler() # data needs to be standardized before PCA; replace MinMaxScaler(), to ensure the mean is zero 

    if testset_ratio == 0: # no testing set, use the whole dataset for training
        X_train, y_train, id_train = X_imputed, y, ids
        X_test_scaled, y_test, id_test = pd.DataFrame(), pd.Series(), pd.Series()

        if trained_scaler is None:
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        else:
            X_train_scaled = pd.DataFrame(trained_scaler.transform(X_train), columns=X_train.columns)

    else: 
        if split_with_ids is not None: # split the dataset based on pre-defined IDs
            idx_train, idx_test = ids.isin(split_with_ids["Train"]), ids.isin(split_with_ids["Test"])
            id_train, id_test = ids[idx_train], ids[idx_test]
            X_train, X_test = X_imputed[idx_train], X_imputed[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]

        else: # split the dataset based on the given ratio
            X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                X_imputed, y, ids, test_size=testset_ratio, random_state=seed)
        
        if trained_scaler is None:
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        else:
            X_train_scaled = pd.DataFrame(trained_scaler.transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(trained_scaler.transform(X_test), columns=X_test.columns)

    return {
        "X_train": X_train_scaled, 
        "X_test": X_test_scaled, 
        "y_train": y_train, 
        "y_test": y_test, 
        "id_train": id_train.tolist() if isinstance(id_train, pd.Series) else id_train, 
        "id_test": id_test.tolist() if isinstance(id_test, pd.Series) else id_test, 
        "scaler": scaler if trained_scaler is None else trained_scaler
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

## Main: ==============================================================================

def main():
    ## Parse command line arguments:
    args = parser.parse_args()

    ## Setup config and constants objects:
    config = Config(args)
    constants = Constants()

    os.makedirs(config.out_folder) # should not exist

    ## Setup logging:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename=config.logging_outpath
    ) # https://zx7978123.medium.com/python-logging-%E6%97%A5%E8%AA%8C%E7%AE%A1%E7%90%86%E6%95%99%E5%AD%B8-60be0a1a6005

    logging.info(f"\nStart at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    ## Copy the current Python script to the output folder:
    shutil.copyfile(
        src=os.path.abspath(__file__), 
        dst=os.path.join(config.out_folder, os.path.basename(__file__))
    )
    logging.info("The current python script is copied to the output folder :-)")

    ## Define the labels and boundaries of age groups:
    age_bin_labels = list(constants.age_groups[config.age_method].keys())
    age_boundaries = list(constants.age_groups[config.age_method].values()) 

    ## Define the labels and boundaries for age correction:
    pad_age_groups = list(constants.age_groups[config.age_correction_groups].keys())
    pad_age_breaks = [ 0 ] + [ x for _, x in list(constants.age_groups[config.age_correction_groups].values()) ] 

    ## Update 'constants.domain_approach_mapping' if necessary:
    if args.include_all_mappings:
        logging.info("Include the 'ALL' domain-approach mapping.")
        constants.domain_approach_mapping["ALL"] = {
            "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
            "approaches": ["MRI", "BEH", "EEG"]
        }
    elif args.only_all_mapping:
        logging.info("Only include the 'ALL' domain-approach mapping.")
        constants.domain_approach_mapping = {
            "ALL": {
                "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
                "approaches": ["MRI", "BEH", "EEG"]
            }
        }    

    ## Save the description of the current execution:
    desc = save_description(
        args, config, constants, age_bin_labels, pad_age_groups
    )
    balancing_method = desc["DataBalancingMethod"]
    n_per_balanced_g = desc["NumPerBalancedGroup"]
    included_models = desc["IncludedOptimizationModels"]
        
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
        DF = load_and_modify_data(
            data_file_path=config.data_file_path, 
            inclusion_file_path=config.inclusion_file_path
        )

        ## Make balanced datasets if specified:
        if balancing_method is not None:
            logging.info(f"Making balanced datasets using '{balancing_method}' method ...")
            target_col, DF_prepared = make_balanced_dataset(
                DF=copy.deepcopy(DF), 
                balancing_method=balancing_method, 
                age_bin_dict=constants.age_groups["wais_8_seg"], 
                n_per_balanced_g=n_per_balanced_g, 
                seed=config.seed
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

    ## If user only wants to prepare the data:
    if args.prepare_data_and_exit:
        logging.info("Data preparation is completed. Exiting the program ...")
        return

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
    train_ids, test_ids = [], []

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
                split_with_ids=config.split_with_ids, 
                testset_ratio=config.testset_ratio, 
                trained_scaler=trained_scaler, 
                seed=config.seed
            ) # a dictionary of train-test splited data, storing the standardized feature values, age, and ID numbers of participants.

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

    logging.info("Preprocessing of all data subsets is completed.")

    logging.info("Starting loop through all groups and orientations ...")    
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
                                    model_names=included_models, 
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


