#!/usr/bin/python

import re
# from collections import defaultdict
import numpy as np
import pandas as pd
import shap 
# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform
from factor_analyzer import Rotator
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

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
        GM_area2lobe_lower = { k.lower(): v for k, v in self._GM_areas_to_lobes().items() }
        
        for feature in included_features:
            if feature.startswith("STRUCTURE"):
                category, hemi, area, measure = feature.split("_")[3::]
                if category == "NULL":
                    f = f"STR_{category}_{measure}"
                else:
                    if (category == "GM") and (measure != "FA"): 
                        lobe = GM_area2lobe_lower[area.lower()]

                        if lobe == "I-C":
                            f = f"STR_{category}_{lobe}_{measure}"
                        else:
                            f = f"STR_{category}_{hemi[0]}_{lobe}_{measure}"
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

    def _GM_areas_to_lobes(self):
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

class FeatureSelector:
    def __init__(self, method, thresh_method, threshold, explained_ratio, max_feature_num, seed, n_jobs=16):
        self.method = method
        self.seed = seed
        self.thresh_method = thresh_method
        self.threshold = threshold
        self.explained_ratio = explained_ratio
        self.max_feature_num = max_feature_num
        self.min_feature_num = 1 # 10
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
        ## Check input:
        assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
        assert isinstance(y, pd.Series) or isinstance(y, np.ndarray), "y should be a pandas Series or a numpy array."
        assert X.shape[0] == y.shape[0], "X and y should have the same number of rows."
        assert (not np.any(np.isnan(X))) and (not np.any(np.isnan(y))), "X and y should not contain NaN values."
        assert (not np.any(np.isinf(X))) and (not np.any(np.isinf(y))), "X and y should not contain infinite values."

        ## Estimate feature importance:
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

        elif self.method.endswith("SHAP"): 

            if self.method.startswith("ElaNet"): # ElasticNet-SHAP
                model = ElasticNetCV(
                    l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 
                    cv=5, random_state=self.seed, n_jobs=self.n_jobs
                ).fit(X, y)
                explainer = shap.Explainer(
                    model=model.predict, 
                    masker=X, # pass a background data matrix instead of a function
                    algorithm="linear"
                )
                shap_values = explainer(X)

            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=.2, random_state=self.seed
                )
                if self.method.startswith("RF"): # RF-SHAP
                    shap_values = shap.TreeExplainer(
                        model=RandomForestRegressor(
                            random_state=self.seed, n_jobs=self.n_jobs
                        ).fit(X_train, y_train), 
                    ).shap_values(X_test)

                elif self.method.startswith("LGBM"): # LGBM-SHAP
                    shap_values = shap.TreeExplainer(
                        model=LGBMRegressor(
                            max_depth=3, min_child_samples=5, 
                            random_state=self.seed, n_jobs=self.n_jobs
                        ).fit(X_train, y_train)
                    ).shap_values(X_test)

            importances = np.abs(shap_values).mean(axis=0)

        ## Fallback:
        if np.isnan(importances).any() or np.all(importances == 0):
            raise ValueError("SHAP values are invalid (NaN or all zero).")

        ## Selection:
        feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        
        if self.thresh_method == "explained_ratio":
            normed_feature_imp = feature_importances / feature_importances.abs().sum()
            cumulative_importances = normed_feature_imp.abs().cumsum()
            num_features = np.argmax(cumulative_importances > self.explained_ratio) + 1
            selected_feature_imp = feature_importances.head(num_features)

        elif self.thresh_method == "fixed_threshold":
            selected_feature_imp = feature_importances[feature_importances.abs() > self.threshold]

        if self.max_feature_num is not None:
            selected_feature_imp = selected_feature_imp.head(self.max_feature_num)
        
        ## Fallback:
        if len(selected_feature_imp) < self.min_feature_num:
            raise ValueError("Too few features are selected. Threshold may be too strict.")

        ## Record:
        self.feature_importances = selected_feature_imp
        self.selected_features = list(selected_feature_imp.index)

        return self
    
    def transform(self, X):
        return X[self.selected_features]