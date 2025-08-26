#!/usr/bin/python

import os
import re
import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC 
from sdv.metadata import Metadata
from rdt.transformers.numerical import FloatFormatter
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

class DataManager:
    def __init__(self, raw_data_path, inclusion_data_path):
        self.raw_data_path = raw_data_path
        self.inclusion_data_path = inclusion_data_path
        self.DF = None
        self.DF_balanced = None

    def load_data(self):
        DF = self._load_from_file()
        DF = self._drop_unnecessary_cols(DF)
        DF = self._convert_to_mean_cols(DF)
        self.DF = DF

        return self

    def make_data_balanced(self, method, balancing_groups, age_bin_dict, balance_to_num, seed, out_dir=None):
        self.balancing_method = method
        self.balancing_groups = balancing_groups
        self.balancing_age_bins = age_bin_dict
        self.balance_to_num = balance_to_num
        self.seed = seed
        DF_ = self.DF.copy(deep=True) # avoid modifying the original dataframe
        DF_ = self._divide_into_groups(DF_) # create column "AGE-GROUP_SEX" 
        DF_imputed = self._fill_missing_by_group(DF_)
        DF_balanced = self._balancing_each_group(DF_imputed, out_dir=out_dir)
        self.DF_balanced = DF_balanced
        
        return self

    def _load_from_file(self):
        '''
        Load data and inclusion table, and merge them to apply inclusion criteria.
        '''
        raw_data = pd.read_csv(self.raw_data_path)
        raw_data.rename(columns={"BASIC_INFO_ID": "ID"}, inplace=True) # ensure consistent ID column name

        inclusion_data = pd.read_csv(self.inclusion_data_path)
        inclusion_data = inclusion_data.query("MRI == 1") # only include participants with MRI data

        DF = pd.merge(raw_data, inclusion_data[["ID"]], on="ID", how='inner') # apply inclusion criteria
        DF.dropna(subset=["ID", "BASIC_INFO_AGE", "BASIC_INFO_SEX"], inplace=True) # should not be missing

        return DF
    
    def _drop_unnecessary_cols(self, DF):
        DF.drop(columns=[
            col for col in DF.columns 
            if ( "RESTING" in col )
            or ( col.startswith("BASIC_") and (not col.startswith("BASIC_Q_")) and (col not in ["BASIC_INFO_AGE", "BASIC_INFO_SEX"]) )
            or ( col.startswith("LANGUAGE_SPEECHCOMP_BEH") and col.endswith("RT") )
            or ( col.startswith("MOTOR_GOFITTS_EEG") and (("Diff" in col) or ("Slope" in col)) )
            or ( col.startswith("MEMORY_EXCLUSION_BEH") and any( kw in col for kw in ["TarMiss", "NewFA", "NonTarFA_PROPORTION", "C2NonTarFA_RT", "C3NonTarFA_RT", "C1NewCR_PROPORTION", "C1NewCR_RTvar"] ) )
            or ( col.startswith("MEMORY_EXCLUSION_EEG") and any( kw in col for kw in ["TarHitNewCRdiff", "NonTarCRNewCRdiff"] ) )
            or ( col.startswith("MEMORY_OSPAN_EEG") and any( kw in col for kw in ["150To350", "AMPLITUDE"] ) )
            or ( col.startswith("MEMORY_MST_MRI") and any( kw in col for kw in ["OldCorSimCorDiff", "OldCorNewCorDiff", "SimCorNewCorDiff", "MD"] ) )
        ] + [
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

        return DF

    def _convert_to_mean_cols(self, DF):
        for col in DF.columns:
            if re.match(r"MOTOR_GFORCE_MRI_.*?HighForce_.*?H_.*?_[a-zA-Z]+", col):
                pair = col.replace("High", "Low")
                DF.insert(loc=DF.columns.get_loc(col), column=col.replace("High", "Mean"), value=DF[[col, pair]].mean(axis=1))
                DF.drop(columns=[col, pair], inplace=True)

        return DF

    def _divide_into_groups(self, DF):
        '''
        Assign "AGE-GROUP_SEX" labels and drop intermediate products.
        '''
        age_bin_edges = [ 0 ] + [ x for _, x in list(self.balancing_age_bins.values()) ] 
        age_bin_labels = list(self.balancing_age_bins.keys())
        DF["AGE-GROUP"] = pd.cut(x=DF["BASIC_INFO_AGE"], bins=age_bin_edges, labels=age_bin_labels) # (min, max]
        DF["SEX"] = DF["BASIC_INFO_SEX"].map({1: "M", 2: "F"})
        DF["AGE-GROUP_SEX"] = DF.loc[:, ["AGE-GROUP", "SEX"]].agg("_".join, axis=1)
        DF.drop(columns=["AGE-GROUP", "SEX"], inplace=True)

        return DF
    
    def _fill_missing_by_group(self, DF, group_col="AGE-GROUP_SEX", fill_with="median"):
        group_names = list(DF[group_col].unique())
        imputed_df_list = []
        for g in group_names:
            sub_df = DF[DF[group_col] == g].copy(deep=True)
            sub_df.set_index("ID", inplace=True)
            sub_X = sub_df.drop(columns=["BASIC_INFO_AGE", "BASIC_INFO_SEX", group_col]) # those should not be imputed
            imputer = SimpleImputer(strategy=fill_with)
            imputed_sub_df = pd.merge(
                sub_df.loc[:, ["BASIC_INFO_AGE", "BASIC_INFO_SEX", group_col]], # add back
                pd.DataFrame(data=imputer.fit_transform(sub_X), columns=sub_X.columns, index=sub_X.index), 
                left_index=True, right_index=True
            )
            imputed_sub_df.reset_index(inplace=True)
            imputed_sub_df.rename(columns={"index": "ID"}, inplace=True)
            imputed_df_list.append(imputed_sub_df)

        DF_imputed = pd.concat(imputed_df_list)
        DF_imputed.reset_index(drop=True, inplace=True)

        return DF_imputed
    
    def _balancing_each_group(self, DF, group_col="AGE-GROUP_SEX", out_dir=None):
        group_names = list(DF[group_col].unique())
        float_cols = [ c for c in DF.columns if c not in ["ID", "BASIC_INFO_AGE", "BASIC_INFO_SEX", group_col] ]

        balanced_data_outpath = os.path.join(out_dir, "balanced_dataset.csv")
        marked_data_outpath = balanced_data_outpath.replace(".csv", " (marked).csv")

        if self.balancing_method in ["CTGAN", "TVAE"]:
            DF_balanced_list = []
            skip_diagnostic = False
            skip_quality_check = False

            for g in group_names:
                sub_balanced_data_outpath = balanced_data_outpath.replace(".csv", f"_{g}.csv")
                
                if os.path.exists(sub_balanced_data_outpath):
                    print(f"Balanced dataset for {g} exists, no need to generate again.")
                    sub_df_balenced = pd.read_csv(sub_balanced_data_outpath)
                    DF_balanced_list.append(pd.read_csv(sub_balanced_data_outpath))

                else:
                    sub_df = DF[DF[group_col] == g]
                    sub_X = sub_df.drop(columns=["ID", group_col])

                    ## Setup metadata:
                    metadata_outpath = os.path.join(out_dir, f"metadata_{g}.json")
                    if os.path.exists(metadata_outpath):
                        metadata = Metadata.load_from_json(metadata_outpath)
                        print(f"Metadata is loaded from {metadata_outpath}")
                    else:
                        metadata = Metadata.detect_from_dataframe(
                            data=sub_X, infer_sdtypes=False, infer_keys=None
                        )
                        metadata.update_columns(
                            column_names=float_cols, sdtype='numerical', computer_representation='Float'
                        )
                        metadata.update_columns_metadata(column_metadata={
                            "BASIC_INFO_SEX": {"sdtype": "categorical"},
                            "BASIC_INFO_AGE": {"sdtype": "numerical", "computer_representation": "Int64"}
                        })
                        metadata.save_to_json(metadata_outpath)
                        print(f"Metadata is saved to {metadata_outpath}")

                    ## Prepare synthesizer:
                    synthesizer_outpath = os.path.join(out_dir, f"synthesizer_{self.balancing_method}_{g}.pkl")
                    if os.path.exists(synthesizer_outpath):
                        if self.balancing_method == "CTGAN":
                            synthesizer = CTGANSynthesizer.load(synthesizer_outpath)
                        else: # TVAE
                            synthesizer = TVAESynthesizer.load(synthesizer_outpath)
                        print(f"Synthesizer is loaded from {synthesizer_outpath}")
                    else:
                        if self.balancing_method == "CTGAN":
                            synthesizer = CTGANSynthesizer(metadata=metadata, verbose=True)
                        else: # TVAE
                            synthesizer = TVAESynthesizer(metadata=metadata, verbose=True)

                        synthesizer.auto_assign_transformers(sub_X)
                        for col in float_cols:
                            synthesizer.update_transformers(column_name_to_transformer={
                                col: FloatFormatter(learn_rounding_scheme=True)
                            })
                        synthesizer.fit(sub_X)
                        synthesizer.save(synthesizer_outpath)
                        print(f"Synthesizer is saved to {synthesizer_outpath}")

                    ## Generate synthetic data:
                    syn_X = synthesizer.sample(num_rows=self.balance_to_num - len(sub_X))

                    ## Run diagnostics:
                    if skip_diagnostic:
                        print("Skipping diagnostic report ...")
                    else:
                        diagnostics_outpath = os.path.join(out_dir, f"diagnostics_{self.balancing_method}_{g}.png")
                        if os.path.exists(diagnostics_outpath):
                            print("Removing old diagnostic report ...")
                            os.remove(diagnostics_outpath)
                        diagnostic = DiagnosticReport()
                        diagnostic.generate(
                            real_data=sub_X, 
                            synthetic_data=syn_X, 
                            metadata=metadata.to_dict()["tables"]["table"]
                        )
                        diagnostic.save(filepath=diagnostics_outpath)
                        print(f"Diagnostic report is saved to {diagnostics_outpath}")

                    ## Perform quality check:
                    if skip_quality_check:
                        print("Skipping quality check ...")
                    else:
                        quality_check_outpath = os.path.join(out_dir, f"quality_check_{self.balancing_method}_{g}.png")
                        if os.path.exists(quality_check_outpath):
                            print("Removing old quality report ...")
                            os.remove(quality_check_outpath)
                        quality = QualityReport()
                        quality.generate(
                            real_data=sub_X, 
                            synthetic_data=syn_X, 
                            metadata=metadata.to_dict()["tables"]["table"]
                        )
                        quality.save(filepath=quality_check_outpath)
                        print(f"Quality check report is saved to {quality_check_outpath}")

                    ## Finally:
                    syn_X.insert(1, "R_S", "Synthetic") 
                    sub_X.insert(1, "R_S", "Real") 
                    sub_df_balenced = pd.concat([sub_X, syn_X], axis=0).copy()
                    sub_df_balenced.to_csv(sub_balanced_data_outpath, index=False)
                
                DF_balanced_list.append(sub_df_balenced)

            DF_balanced = pd.concat(DF_balanced_list).reset_index(drop=True)
            DF_balanced.insert(0, "ID", [ f"sub-{x:04d}" for x in DF_balanced.index ])
            DF_balanced.to_csv(marked_data_outpath, index=False)
            print(f"Balanced dataset is saved to {marked_data_outpath}")

        else:
            if self.balancing_method == "SMOTENC":
                sampler = SMOTENC(
                    categorical_features=["BASIC_INFO_AGE", "BASIC_INFO_SEX"], 
                    sampling_strategy={ g: self.balance_to_num for g in group_names },
                    random_state=self.seed
                )
            elif self.balancing_method == "downsample":
                sampler = RandomUnderSampler(
                    sampling_strategy={ g: self.balance_to_num for g in group_names },
                    random_state=self.seed, 
                    replacement=False
                )
            elif self.balancing_method == "bootstrap":
                sampler = RandomUnderSampler(
                    sampling_strategy={ g: self.balance_to_num for g in group_names },
                    random_state=self.seed, 
                    replacement=True
                )
            else:
                raise NotImplementedError(f"Balancing method {self.balancing_method} is not implemented.")
            
            X_resampled, y_resampled = sampler.fit_resample(
                X=DF.drop(columns=["ID", group_col]), 
                y=DF[group_col].astype(str)
            )
            DF_balanced = pd.merge(
                pd.DataFrame({group_col: y_resampled}), X_resampled, 
                left_index=True, right_index=True
            )
            DF_balanced.insert(0, "ID", [ f"sub-{x:04d}" for x in DF_balanced.index ])

            DF_balanced = self._mark_synthetic_data(DF, DF_balanced)
            DF_balanced.to_csv(marked_data_outpath, index=False)
            print(f"Balanced dataset is saved to {marked_data_outpath}")

        DF_balanced.drop(columns=["R_S"], inplace=True)
        DF_balanced.to_csv(balanced_data_outpath, index=False)

        return DF_balanced
    
    def _mark_synthetic_data(self, DF, DF_upsampled): 
        ## Find the common NA-free columns between the two dataframes:
        DF_nona = DF.dropna(axis=1)
        nona_cols = list(DF_nona.columns)
        common_nona_cols = [ x for x in nona_cols if x in DF_upsampled.columns ]
        common_nona_cols.remove("ID")

        ## Use inner join on common_nona_cols to find real data:
        DF_real = ( 
            DF_upsampled
            .loc[:, common_nona_cols]
            .reset_index() # get index in the upsampled dataframe
            .merge(DF.loc[:, common_nona_cols], how='inner', on=common_nona_cols)
            .set_index('index') # index back
        )

        ## Add a new column to mark whether the data is real or synthetic:
        DF_upsampled.insert(1, "R_S", "Synthetic")
        DF_upsampled.loc[DF_real.index, "R_S"] = "Real"

        return DF_upsampled