import lightgbm as lgb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import plot_tree
from xgboost import XGBRegressor, XGBClassifier

from catboost import Pool, CatBoostRegressor, CatBoostClassifier

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, f1_score, mean_absolute_error, mean_squared_error # fmt: skip

from scipy.stats import kendalltau, pearsonr, spearmanr

import shap

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import ElasticNet

from gbt_plot_utils import *

from SALib.analyze import sobol


class GenGBT:
    def __init__(
        self, Q, QoI, ts=None, threshold=None, model_type="regression", gbt_method="xgboost", splitpercent=0.80, splittime=None, split_random=True, important=None):  # fmt: skip
        
        self.Q = Q
        self.cols = self.Q.param_names()
        self.label = QoI
        self.method = gbt_method
        self.model_type = model_type
        # self.label = label
        self.threshold = threshold
        # # categorical and numerical columns
        # self.catcols = list(d[cols].select_dtypes(include='category').columns)
        # self.catcols_ind = d[cols].columns.get_indexer(self.catcols)
        # self.numcols = list(d[cols].select_dtypes(exclude='category').columns)

        # self.df = d
        # self.cols = cols
        # # Model dependent data preparation
        # d, cols = self.model_dependent_data_preparation(d, cols, self.catcols)
        self.important = important

        # # Split test and training data
        # self.X_train, self.X_test, self.y_train, self.y_test = self.split_test_and_train_data(
        #     d, cols, label, threshold, split_random=split_random, splitpercent=splitpercent, splittime=splittime, ts=ts,) # fmt: skip

        # if resampling:
        #     if self.model_type == "regression":
        #         print(
        #             "RESAMPLING is only available for classification problem, problem will be handled without resampling")
        #         # showwarning("RESAMPLING is only available for classification problem, problem will be handled without resampling")
        #     else:
        #         ros = RandomOverSampler(random_state=0)
        #         print("Original size of train and test samples :", self.y_train.shape, self.y_test.shape)
        #         print("Original ratio of trues in the training set", np.sum(self.y_train) / self.y_train.shape[0] * 100,
        #               "%")
        #         print("Original ratio of trues in the testing set", np.sum(self.y_test) / self.y_test.shape[0] * 100,
        #               "%")
        #         self.X_train, self.y_train = ros.fit_resample(self.X_train, self.y_train)
        #         self.X_test, self.y_test = ros.fit_resample(self.X_test, self.y_test)
        #         print("Size of train and test samples after resampling:", self.y_train.shape, self.y_test.shape)
        #         print("Ratio of trues in the training set after resampling",
        #               np.sum(self.y_train) / self.y_train.shape[0] * 100, "%")
        #         print("Ratio of trues in the testing set after resampling",
        #               np.sum(self.y_test) / self.y_test.shape[0] * 100, "%")

    def model_dependent_data_preparation(self, d, cols, catcols, method=None):
        if method == None:
            method = self.method

        # Model dependent data preparation
        if method == "scikit" or method == "xgboost":
            # One hot encoding
            # Features to be trained without any categorical ones
            extended_cols = list(set(cols) - set(catcols))
            # One-hot encoding
            if catcols != []:
                ohe = pd.get_dummies(d[catcols])
                extended_cols.extend(list(ohe.columns))
                # Join dataframes
                d = pd.concat([d, ohe], axis=1)
                cols = extended_cols
                # Fill Nans
                d[cols] = d[cols].fillna(0)
                # Change name of categorical features to ohe feature names
                self.catcols = list(ohe.columns)
        elif method == "catboost":
            # convert categories to strings
            for i_col in catcols:
                d[i_col] = d[i_col].astype(str).astype('category')

        return d, cols

    def split_test_and_train_data(self, d, cols, label, threshold, splitpercent, ts=None, splittime=None,
                                  resampling=False, split_random=True, verbose_flag=False):
        if split_random == False:
            if splittime == None:
                split_row = int(d.shape[0] * splitpercent)
                splittime = d[ts].iloc[split_row]
            X_train = d[(d[ts] < splittime) & (d[label] == d[label])][cols]
            X_test = d[(d[ts] >= splittime) & (d[label] == d[label])][cols]
            if threshold == None:  # regression
                y_train = d[(d[ts] < splittime) & (d[label] == d[label])][label]
                y_test = d[(d[ts] >= splittime) & (d[label] == d[label])][label]
            else:  # classification
                y_train = d[(d[ts] < splittime) & (d[label] == d[label])][label] > threshold
                y_test = d[(d[ts] >= splittime) & (d[label] == d[label])][label] > threshold
        else:
            X_train, X_test, y_train, y_test = train_test_split(d[cols], d[label], test_size=1 - splitpercent,
                                                                random_state=42)
        return X_train, X_test, y_train, y_test


    def train_model(self, d_X, d_y, n_est=150, max_d=3, learning_rate=0.15, k_fold=None, num_leaves=None, 
                    plot_flag=True, ts=None, splitpercent=0.80, splittime=None, split_random=True,
                    task_type='GPU', verbose_flag=False):
              
        """
        Handles the training of the model. Supports k-fold cross-validation.
        """
        if type(d_X) != pd.DataFrame:
            d_X = pd.DataFrame(d_X)
        if type(d_y) != pd.DataFrame:
            d_y = pd.DataFrame(d_y)
        if k_fold:
            # Perform k-fold cross-validation
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            fold = 1
            fold_results = []
            for train_indices, val_indices in kf.split(d_X):
                print(f"Training Fold {fold}/{k_fold}...")
                X_train, X_val = d_X.iloc[train_indices], d_X.iloc[val_indices]
                y_train, y_val = d_y.iloc[train_indices], d_y.iloc[val_indices]
                
                # Call build_model for each fold
                model = self.build_model(X_train, y_train, n_est, max_d, learning_rate, 
                                         task_type, verbose_flag)
                
                # Validate the model on the validation set
                predictions = model.predict(X_val)
                if self.model_type == "regression":
                    mse = mean_squared_error(y_val, predictions)
                    print(f"Fold {fold} MSE: {mse:.4f}")
                    fold_results.append(mse)
                # Extend with classification evaluation if needed
                fold += 1
            
            # Report average performance
            print(f"Average MSE across folds: {np.mean(fold_results):.4f}")
        else:
            # Default 80-20 split
            split_index = int(len(d_X) * 0.8)
            X_train, X_val = d_X.iloc[:split_index], d_X.iloc[split_index:]
            y_train, y_val = d_y.iloc[:split_index], d_y.iloc[split_index:]
            
            print("Training with default 80-20 split...")
            model = self.build_model(X_train, y_train, n_est, max_d, learning_rate, num_leaves, plot_flag, 
                    task_type, verbose_flag, ts, splitpercent, splittime, split_random)
            
            predictions = model.predict(X_val)
            if self.model_type == "regression":
                mse = mean_squared_error(y_val, predictions)
                print(f"Validation MSE: {mse:.4f}")
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        
        return model
      
    def train_and_validate(self, X_train, y_train, X_val, y_val, num_of_iter=150, max_depth=3, learning_rate=0.15, num_leaves=None, plot_flag=True, task_type='GPU', verbose_flag=True, ts=None, splitpercent=0.80, splittime=None, split_random=True):
        max_d = max_depth
        n_est = num_of_iter
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Train model
        if self.method == 'scikit':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                                  max_leaf_nodes=num_leaves)
            elif self.model_type == "classification":  # classification
                model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                                   max_leaf_nodes=num_leaves)
            # fit model
            model.fit(X_train, y_train)

        elif self.method == 'xgboost':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                if num_leaves == None:
                    model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=n_est,
                                         max_depth=max_d)
                else:
                    model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=n_est,
                                         max_leaves=num_leaves, grow_policy='lossguide')
            elif self.model_type == "classification":  # classification
                if num_leaves == None:
                    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                          max_leaf_nodes=num_leaves)
                else:
                    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_est, max_leaf_nodes=num_leaves,
                                          grow_policy='lossguide')
            # fit model
            #model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=verbose_flag)
            model.fit(X_train, y_train, verbose=verbose_flag)

        elif self.method == 'catboost':
            cols = list(X_train.columns)
            self.catcols_ind = []
            pool_train = Pool(X_train, y_train, cat_features=self.catcols_ind, feature_names=cols)
            pool_test = Pool(X_val, y_val, cat_features=self.catcols_ind, feature_names=cols)
            # Initialize CatBoostClassifier/Regressor
            if self.model_type == "regression":  # regression
                if num_leaves == None:
                    model = CatBoostRegressor(max_depth=max_d, learning_rate=learning_rate, n_estimators=n_est,
                                              loss_function="RMSE", eval_metric="MAE", use_best_model=True, verbose=0,
                                              task_type=task_type)
                else:
                    model = CatBoostRegressor(learning_rate=learning_rate, grow_policy='Lossguide', max_depth=max_d,
                                              n_estimators=n_est, max_leaves=num_leaves, loss_function="RMSE",
                                              eval_metric="MAE", use_best_model=True, verbose=0, task_type=task_type)
            elif self.model_type == "classification":  # classification
                if num_leaves == None:
                    model = CatBoostClassifier(max_depth=max_d, learning_rate=learning_rate, n_estimators=n_est,
                                               loss_function="Logloss", use_best_model=True, verbose=0,
                                               task_type=task_type)
                else:
                    model = CatBoostClassifier(learning_rate=learning_rate, grow_policy='Lossguide', max_depth=max_d,
                                               n_estimators=n_est, max_leaves=num_leaves, loss_function="Logloss",
                                               use_best_model=True, verbose=0, task_type=task_type)
            # Fit model
            model.fit(pool_train, eval_set=pool_test, plot=plot_flag, verbose=verbose_flag)

        elif self.method == "lightgbm":
            assert not any(' ' in x for x in X_train.columns), 'No space allowed in column names with LightGBM.' 
            pool_train = lgb.Dataset(X_train, label=y_train)  # , categorical_feature=categorical)
            pool_test = lgb.Dataset(X_val, label=y_val)
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                hyper_params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'boosting': 'gbdt',
                    'max_depth': max_d,
                    'n_estimators': n_est,
                    'num_leaves': num_leaves,
                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.5,
                    'bagging_freq': 20,
                    'learning_rate': learning_rate,
                    'verbose': 0
                }
            elif self.model_type == "classification":  # classification
                hyper_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': num_leaves,
                    'boosting': 'gbdt',
                    'max_depth': max_d,
                    'n_estimators': n_est,
                    'learning_rate': learning_rate,
                    'verbose': 0
                }
            else:
                raise ValueError
            model = lgb.train(hyper_params, pool_train, valid_sets=pool_test, num_boost_round=5000,
                              callbacks=[lgb.early_stopping(stopping_rounds=50)])
        elif self.method == 'ElasticNet':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                model = ElasticNet().fit(X_train, y_train)
            elif self.model_type == "classification":  # classification
                raise ('Use ElasticNet with regression or logistic regression!')
        elif self.method == "log_reg":
            model = self.build_lreg_model()
        else:
            raise ('Unknown method: ', self.method)
        self.model = model   
        self.prediction = model.predict
        y_tr_pred = self.predict(X_train)
        y_vl_pred = self.predict(X_val)
        mse_tr = mean_squared_error(y_train, y_tr_pred)
        mse_vl = mean_squared_error(y_val, y_vl_pred)
        return mse_tr, mse_vl
    
    
    def build_model(self, X_train, y_train, num_iter=150, max_depth=3, learning_rate=0.15, num_leaves=None, plot_flag=True, 
                    task_type='GPU', verbose_flag=False, ts=None, splitpercent=0.80, splittime=None, split_random=True):
        max_d = max_depth
        n_est = num_iter
        
        # d = pd.concat([d_X, d_y], axis=1)
        # self.df = d
                
        # # Split test and training data
        # self.X_train, self.X_test, self.y_train, self.y_test = self.split_test_and_train_data(
        #     d, self.cols, self.label, threshold=self.threshold, split_random=split_random, splitpercent=splitpercent, splittime=splittime, ts=ts) # fmt: skip
        
        # Train model
        if self.method == 'scikit':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                                  max_leaf_nodes=num_leaves)
            elif self.model_type == "classification":  # classification
                model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                                   max_leaf_nodes=num_leaves)
            # fit model
            model.fit(self.X_train, self.y_train)

        elif self.method == 'xgboost':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                if num_leaves == None:
                    model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=n_est,
                                         max_depth=max_d)
                else:
                    model = XGBRegressor(objective="reg:squarederror", learning_rate=learning_rate, n_estimators=n_est,
                                         max_leaves=num_leaves, grow_policy='lossguide')
            elif self.model_type == "classification":  # classification
                if num_leaves == None:
                    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_est, max_depth=max_d,
                                          max_leaf_nodes=num_leaves)
                else:
                    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_est, max_leaf_nodes=num_leaves,
                                          grow_policy='lossguide')
            # fit model
            #model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=verbose_flag)
            model.fit(X_train, y_train, verbose=verbose_flag)

        elif self.method == 'catboost':
            cols = list(self.X_train.columns)
            self.catcols_ind = []
            pool_train = Pool(self.X_train, self.y_train, cat_features=self.catcols_ind, feature_names=cols)
            pool_test = Pool(self.X_test, self.y_test, cat_features=self.catcols_ind, feature_names=cols)
            # Initialize CatBoostClassifier/Regressor
            if self.model_type == "regression":  # regression
                if num_leaves == None:
                    model = CatBoostRegressor(max_depth=max_d, learning_rate=learning_rate, n_estimators=n_est,
                                              loss_function="RMSE", eval_metric="MAE", use_best_model=True, verbose=0,
                                              task_type=task_type)
                else:
                    model = CatBoostRegressor(learning_rate=learning_rate, grow_policy='Lossguide', max_depth=max_d,
                                              n_estimators=n_est, max_leaves=num_leaves, loss_function="RMSE",
                                              eval_metric="MAE", use_best_model=True, verbose=0, task_type=task_type)
            elif self.model_type == "classification":  # classification
                if num_leaves == None:
                    model = CatBoostClassifier(max_depth=max_d, learning_rate=learning_rate, n_estimators=n_est,
                                               loss_function="Logloss", use_best_model=True, verbose=0,
                                               task_type=task_type)
                else:
                    model = CatBoostClassifier(learning_rate=learning_rate, grow_policy='Lossguide', max_depth=max_d,
                                               n_estimators=n_est, max_leaves=num_leaves, loss_function="Logloss",
                                               use_best_model=True, verbose=0, task_type=task_type)
            # Fit model
            model.fit(pool_train, eval_set=pool_test, plot=plot_flag, verbose=verbose_flag)

        elif self.method == "lightgbm":
            assert not any(' ' in x for x in self.X_train.columns), 'No space allowed in column names with LightGBM.' 
            pool_train = lgb.Dataset(self.X_train, label=self.y_train)  # , categorical_feature=categorical)
            pool_test = lgb.Dataset(self.X_test, label=self.y_test)
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                hyper_params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'boosting': 'gbdt',
                    'max_depth': max_d,
                    'n_estimators': n_est,
                    'num_leaves': num_leaves,
                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.5,
                    'bagging_freq': 20,
                    'learning_rate': learning_rate,
                    'verbose': 0
                }
            elif self.model_type == "classification":  # classification
                hyper_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': num_leaves,
                    'boosting': 'gbdt',
                    'max_depth': max_d,
                    'n_estimators': n_est,
                    'learning_rate': learning_rate,
                    'verbose': 0
                }
            else:
                raise ValueError
            model = lgb.train(hyper_params, pool_train, valid_sets=pool_test, num_boost_round=5000,
                              callbacks=[lgb.early_stopping(stopping_rounds=50)])
        elif self.method == 'ElasticNet':
            # Initialize Classifier/Regressor
            if self.model_type == "regression":  # regression
                model = ElasticNet().fit(self.X_train, self.y_train)
            elif self.model_type == "classification":  # classification
                raise ('Use ElasticNet with regression or logistic regression!')
        elif self.method == "log_reg":
            model = self.build_lreg_model()
        else:
            raise ('Unknown method: ', self.method)
        self.model = model
        return model

    def predict(self, X_test, **kwargs):
        #pred = self.model.predict(X_test)
        pred = self.prediction(X_test)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred
    
    def evaluate_model(self, X_test, y_test, verbose=False, opt_thresh_comp="from_ROC"):
        if self.model_type == "classification":  # For classification problem

            if self.method == "lightgbm":
                pred = self.model.predict(X_test)
            else:
                pred = self.model.predict_proba(X_test)[:, 1]

            # Get optimal value of threshold
            fpr, tpr, thresh_ROC = roc_curve(y_test, pred)
            precision, recall, thresh_prec = precision_recall_curve(y_test, pred)

            if opt_thresh_comp == "from_ROC":
                opt_thresh = self.compute_optimal_threshold_from_ROC(tpr, fpr, thresh_ROC)
            elif opt_thresh_comp == "from_prec":
                opt_thresh = self.compute_optimal_threshold_from_prec_recall(precision, recall, thresh_prec)
            else:
                raise ("There is no optimal threshold method {}. Possible options are from_ROC or from_prec".format(
                    opt_thresh_comp))

            # Get predicted classes

            pred_class = pred > opt_thresh

            # if self.method == "catboost":
            #    pred_class = pred_class == "True"
            # evaluate model
            model_eval = {
                "auc": roc_auc_score(y_test, pred),
                "f1_score": f1_score(np.array(y_test), pred_class)
            }

            if verbose:
                plot_ROC_and_recall_curve = True
                plot_conf_matrix = True
                print(model_eval)
            else:
                plot_ROC_and_recall_curve = False
                plot_conf_matrix = False

            if plot_ROC_and_recall_curve:
                # Plot ROC CURVE
                plt.figure(figsize=(10, 3))
                plt.subplot(1, 2, 1)
                plt.plot(fpr, tpr)
                plt.title("ROC curve")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                # Plot recall curve
                plt.subplot(1, 2, 2)
                plt.plot(recall, precision)
                plt.title("Precision-recall curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.show()

            if plot_conf_matrix:
                # Print confusion matrix
                print("Confusion Matrix:")
                conf_matrix = pd.DataFrame(confusion_matrix(y_test, pred_class),
                                           columns=["predicted GOOD quality", "predicted BAD quality"],
                                           index=["actual GOOD quality", "actual BAD quality"])
                display(conf_matrix)


        elif self.model_type == "regression":  # for regression problem
            pred = self.model.predict(X_test)
            model_eval = {
                "Kendall_tau": kendalltau(y_test, pred)[0],
                #"Pearson": pearsonr(self.y_test, pred)[0],
                #"Spearman": spearmanr(self.y_test, pred)[0],
                "MSE": mean_squared_error(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "STD_of_label": self.y_train.std()
            }
            if verbose == True:
                print('Kendall tau correlation - measure of correspondance between two rankings: %.3f' % model_eval["Kendall_tau"])
                #print('Pearson correlation - measure of linear realationship (cov normalised): %.3f' % model_eval["Pearson"])
                #print('Spearman correaltion - cov(rank(y1), rank(y2)/stdv(rank(y1))): %.3f' % model_eval["Spearman"])
                print("mean squared error: ", model_eval["MSE"])
                print("Root mean squared error: ", model_eval["RMSE"]) 
                print("STD of label:", model_eval["STD_of_label"])
                print("MAE: ", model_eval["MAE"])
        # pdb.set_trace()
        df = pd.DataFrame(model_eval, index=[0])
        df['X_train_cols'] = [self.X_train.columns]
        df['label'] = [self.y_train.columns] #name
        return [df, model_eval]

    def compute_optimal_threshold_from_ROC(self, tpr, fpr, thresh):
        J = tpr - fpr
        ix = np.argmax(J)
        ROC_thresh = thresh[ix]
        return ROC_thresh

    def compute_optimal_threshold_from_prec_recall(self, precision, recall, thresh):
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        prec_thresh = thresh[ix]
        return prec_thresh

    def get_global_feature_importances(self, verbose=True, n_imp_feats=30, feats_to_highlight=None):
        # Pandas series with important features and their importances
        cols = list(self.X_train.columns)
        if self.method == "lightgbm":
            global_imp = self.get_gbt_feature_importance(self.model.feature_importance(importance_type='gain'))
        else:
            global_imp = self.get_gbt_feature_importance(self.model.feature_importances_)
        # Important features
        imp_feats = global_imp.index
        # Get most important categorical features
        imp_cat_feats = self.get_most_important_cat_features(imp_feats, n_imp_feats=n_imp_feats, verbose=verbose)
        # Print global importance list
        if verbose:
            self.print_feature_importances(global_imp, "Global feature importances",
                                           feats_to_highlight=feats_to_highlight)
        return global_imp, imp_feats, imp_cat_feats

    def get_most_important_cat_features(self, imp_feats, n_imp_feats=30, verbose=False):
        imp_cat_feats = list(set(imp_feats[:n_imp_feats]).intersection(set(self.catcols)))
        if verbose:
            if (len(imp_cat_feats) > 0):
                print("Categorical features in the top {} importance list:".format(n_imp_feats))
                print(imp_cat_feats)
            else:
                print("There is no categorical feature in the top {} importance list".format(n_imp_feats))

        return imp_cat_feats

    def print_feature_importances(self, imp, title, feats_to_highlight=None):
        print("")
        print(title)
        imp_df = pd.DataFrame()
        imp_df["feature"] = imp.index
        imp_df["importance"] = imp.values
        pd.options.display.max_colwidth = 100
        pd.options.display.max_rows = 200
        if feats_to_highlight == None:
            imp_form = imp_df.style.apply(
                lambda x: ['background-color: lightblue' if x.feature in self.catcols else '' for i in x], axis=1)
            display(imp_form)
        elif isinstance(feats_to_highlight, str):
            imp_form = imp_df.style.apply(lambda x: [
                'background-color: lightblue' if x.feature in self.catcols else 'background-color: orange' if (
                            x.feature in feats_to_highlight) else "" for i in x], axis=1)
            display(imp_form)
        else:  # if feats_to_highlight is a list
            imp_form = imp_df.style.apply(lambda x: ['background-color: orange' if any([x.feature in f for f in
                                                                                        feats_to_highlight]) else 'background-color: lightblue' if x.feature in self.catcols else ""
                                                     for i in x], axis=1)
            display(imp_form)

    def get_local_feature_importances(self, q, n=4, plot_flag=False, verbose=False, feats_to_highlight=None):
        shap.initjs()
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(q)
        if isinstance(shap_values, list) and len(
                shap_values) == 2:  # if shap gives an array for both classes (True and False)
            shap_values = shap_values[1]
        local_av_imp = self.get_avarage_shap_values(shap_values)
        local_max_imp = self.get_max_shap_values(shap_values)
        feats_with_top_n_mixed_imp = self.get_feats_with_top_n_mixed_impacts(n, local_av_imp, local_max_imp)
        if plot_flag:
            # Plot summary of avarage effect of the features
            shap.summary_plot(shap_values, q, plot_type="bar", plot_flag=True, matplotlib=True)
            # Plot summary of local effect of the features
            shap.summary_plot(shap_values, self.X_train, plot_flag=True)
            plt.show()
        if verbose:
            self.print_feature_importances(local_av_imp, "Avarage local importances (avarage SHAP values)",
                                           feats_to_highlight=feats_to_highlight)
            self.print_feature_importances(local_max_imp, "Maximum local importances (max SHAP values)",
                                           feats_to_highlight=feats_to_highlight)
            print("Features with top {} highest local and avarage impact:".format(n))
            print(feats_with_top_n_mixed_imp)
        return shap_values, local_av_imp, local_max_imp, feats_with_top_n_mixed_imp
    
    def get_shap_values(self, predict_fn, q, forced=False, explainer_type="treeexplainer"):
        if explainer_type == "treeexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.TreeExplainer(self.model)
                self.explainer = explainer
        shap_values = self.explainer(q)
        #explainer = shap.KernelExplainer(predict_fn, q)
        #shap_values = explainer(q)
        return shap_values
    
    def compute_partial_vars(self, model_obj, max_index):
        paramset = model_obj.Q
        QoI_names = model_obj.QoI_names
        problem = {
            'num_vars': paramset.num_params(), 'names': paramset.param_names(), 'dists': paramset.dist_types, 'bounds': paramset.dist_params
            }
        q = paramset.sample(method='Sobol_saltelli', n=32768) # saltelli working only for uniform distribution
        # https://salib.readthedocs.io/en/latest/user_guide/advanced.html
        q_df = pd.DataFrame(q, columns=paramset.param_names())
        #q = paramset.sample(method='MC', n=1000)
        #xi = model_obj.get_scaled_q(q)      
        #y = model_obj.model.predict(q)
        y = model_obj.predict(q_df)
        #y = model_obj.get_orig_y(y_t)
        
        # Run model
        S1 = []
        S2 = []
        for i in range(y.shape[1]):
            y_i = y[:,i]

            # Sobol analysis
            Si_i = sobol.analyze(problem, y_i)
            T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si_i)
            df = Si_i.to_df()
            cols_S1 = list(df[1].index)
            cols_S2 = list(df[2].index)

            S1.append(first_Si['S1'])
            S2.append(second_Si['S2'])

        S1 = np.array(S1)
        S2 = np.array(S2)

        col_names = cols_S1
        sobol_index = S1
        if max_index == 2:
            sobol_index = np.concatenate([S1, S2], axis=1)
            col_names = cols_S1 + cols_S2
            col_names = [f"{x[0]} {x[1]}" if isinstance(x, tuple) else x for x in col_names]
        
        y_var = np.broadcast_to(y.var(axis=0).reshape(-1,1), sobol_index.shape)
        partial_variance = np.multiply(sobol_index, y_var)
        
        y_var = y.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var
             
        partial_var_df, sobol_index_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names), pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)
        

        return partial_var_df, sobol_index_df, y_var

    def get_avarage_shap_values(self, shap_values):
        cols = list(self.X_train.columns)
        av_shap_values = np.mean(np.abs(shap_values), axis=0)
        local_av_imp = self.get_gbt_feature_importance(av_shap_values)
        return local_av_imp

    def get_max_shap_values(self, shap_values):
        cols = list(self.X_train.columns)
        av_shap_values = np.max(np.abs(shap_values), axis=0)
        local_max_imp = self.get_gbt_feature_importance(av_shap_values)
        return local_max_imp

    def get_feats_with_top_n_mixed_impacts(self, n, local_av_imp, local_max_imp):
        top_n_av = list(local_av_imp.iloc[:n].index)
        top_n_max = list(local_max_imp.iloc[:n].index)
        feats_with_top_n_mixed_imp = list((set(top_n_av)).union(set(top_n_max)))
        return feats_with_top_n_mixed_imp

    def plot_correlation_between_numeric_features(self, feats):
        feats = set(feats) - set(self.catcols)
        d = pd.concat([self.X_train[feats], self.y_train], axis=1)
        corrplot(df=d[feats])

    def plot_trees(self, n_est):
        cols = list(self.X_train.columns)
        # n_est
        if self.method == 'catboost':
            pool_train = Pool(self.X_train, self.y_train, cat_features=self.catcols_ind, feature_names=cols)
            for i in range(n_est):
                print('Tree', i, ':')
                print(self.model.plot_tree(tree_idx=i, pool=pool_train))
        elif self.method == "scikit":
            self.get_gbt_model_rules(self.model)
        elif self.method == "lightgbm":
            # fig, ax = plt.subplots(n_est, figsize=(10, 30))
            for i in range(n_est):
                # lgb.create_tree_digraph(self.model, tree_index=i)
                lgb.plot_tree(self.model, tree_index=i)
        elif self.method =="xgboost":
            fig, ax = plt.subplots(n_est,figsize=(30, 30))
            for i in range(n_est):
                plot_tree(self.model, num_trees=i, ax=ax[i])
            plt.show()

    def scatterplot_gbt(self, feats, threshold, outlier=False, hue=None):
        d = pd.concat([self.X_train, self.y_train], axis=1)
        label = self.y_train.name
        for f in feats:
            scatterplot(d, label, f, outlier, threshold=threshold)
            scatterplot(d, label, f, outlier, threshold=threshold, regressionline=True)


    def get_gbt_model_rules(self, gbt_model, k=None):
        """Export scikit learn GBT model rules to function"""
        if len(self.cols) == 1:
            # duplication of the same feature name is needed if only 1 was used for GBT training!
            column_names = self.cols * 3
        INDENT = str("    ")
        num_of_trees = len(gbt_model.estimators_)
        if k != None:
            num_of_trees = min(k, num_of_trees)

        func_str = ["def score_gbt(eval_df, model_name):"]
        func_str += [INDENT + '"""Score segment by gbt rule"""']
        func_str += []
        func_str += [INDENT + "def score_by_gbt_tree_rule(row):"]
        func_str += [INDENT + '# GBT model generated by Scikit-Learn']
        func_str += [INDENT + INDENT + "score = 0.0", ]
        for i in range(num_of_trees):
            func_str += [INDENT + INDENT + '### tree_%i ###' % (i + 1)]
            func_str += print_tree_with_names(gbt_model.estimators_[i, 0].tree_, self.cols, INDENT)
            func_str += ['']
        func_str += [INDENT + INDENT + "return score"]
        func_str += []
        func_str += [INDENT + "eval_df['SCORE_'+model_name] = eval_df.apply(score_by_gbt_tree_rule, axis=1)"]
        func_str += [INDENT + "return eval_df"]
        for line in func_str:
            print(line)
        return func_str

    def get_gbt_feature_importance(self, importances):
        gbt_all_importances = pd.Series(importances, index=self.X_train.columns, name="feature importance").sort_values(
            ascending=False)
        return gbt_all_importances[gbt_all_importances > 0]

    def build_lreg_model(self, splitpercent=0.66, splittime=None, verbose=True,
                         split_random=True, solver='newton-cg'):
        """important_n = [var+'_n' for var in important]
        if(splittime==None):
            df = d[important+[label]].fillna(0)
            normalize(df,important)
            trg = d[label]<=threshold
            features_train, features_test, labels_train, labels_test = train_test_split(df[important_n], trg, test_size=splitpercent,random_state = 42)
        else:
            df = d[important+[label,'ts']].fillna(0)
            normalize(df,important)
            features_train= df[(df['ts']<splittime) & (df[label]==df[label])][important_n]
            features_test = df[(df['ts']>=splittime) & (df[label]==df[label])][important_n]
            labels_train  = df[(df['ts']<splittime) & (df[label]==df[label])][label]<threshold
            labels_test   = df[(df['ts']>=splittime) & (df[label]==df[label])][label]<threshold
        """
        d = self.df[self.important + [self.label, 'ts']].fillna(0)

        if split_random == False:
            if splittime == None:
                split_row = int(d.shape[0] * splitpercent)
                splittime = d["ts"].iloc[split_row]
            features_train = d[(d['ts'] < splittime) & (d[self.label] == d[self.label])][self.important]
            features_test = d[(d['ts'] >= splittime) & (d[self.label] == d[self.label])][self.important]
            if self.threshold == None:  # regression
                labels_train = d[(d['ts'] < splittime) & (d[self.label] == d[self.label])][self.label]
                labels_test = d[(d['ts'] >= splittime) & (d[self.label] == d[self.label])][self.label]
            else:  # classification
                labels_train = d[(d['ts'] < splittime) & (d[self.label] == d[self.label])][self.label] > self.threshold
                labels_test = d[(d['ts'] >= splittime) & (d[self.label] == d[self.label])][self.label] > self.threshold
        else:
            features_train, features_test, labels_train, labels_test = train_test_split(d[self.important], d[self.label],
                                                                                        test_size=1 - splitpercent,
                                                                                        random_state=42)

        logreg = LogisticRegression(solver=solver)

        logreg.fit(features_train, labels_train)
        pred = logreg.predict_proba(features_test)[:, 1]
        auc = roc_auc_score(labels_test, pred)
        print('Logreg model for', self.label, 'AUC =', auc)
        if verbose:
            pretty_print_df(df=self.get_lreg_coefficients(logreg, self.important))
        return auc

    @staticmethod
    def get_lreg_coefficients(lreg_model, cols, regression=False):
        intercept_df = pd.DataFrame({"name": "(intercept)", "value": lreg_model.intercept_})
        if regression:
            coef_df = pd.DataFrame({"name": cols, "value": lreg_model.coef_}).sort_values(by='value')
        else:
            coef_df = pd.DataFrame({"name": cols, "value": lreg_model.coef_[0]}).sort_values(by='value')
        return intercept_df.append(coef_df, ignore_index=True)
