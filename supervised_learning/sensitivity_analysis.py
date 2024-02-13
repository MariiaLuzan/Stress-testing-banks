import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from supervised_learning.cross_validation import PanelDataSplit
from sklearn.model_selection import cross_val_score


def estimate_root_mean_squared_error(y_test, y_pred, lower_limit = None, upper_limit=None):
    """
    Estimates the root mean squared error excluding outliers, which are defined as all data points that 
    fall above the upper_limit or below the lower_limit.
    
    Args:
    y_test (Numpy Array of shape (n,) or Pandas Series): Actual values of the response variable
    y_pred (Numpy Array of shape (n,) or Pandas Series): Predicted values of the response variable
    lower_limit (Float): Values lower than this limit are treated as outliers and excluded from the calculation
    upper_limit (Float): Values higher than this limit are treated as outliers and excluded from the calculation
        
    Returns:
    root_mean_sq_error (Float): Root mean squared error
    
    """
    y_compare = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred})
    
    if lower_limit!=None:
        y_compare = y_compare[(y_compare['y_test']>lower_limit)&(y_compare['y_test']<upper_limit)]
        
    root_mean_sq_error = mean_squared_error(y_compare['y_test'], y_compare['y_pred'])**0.5    
        
    return root_mean_sq_error


def estimate_root_mean_squared_error_outer(y_test, y_pred, lower_limit = None, upper_limit=None):
    """
    Estimates the root mean squared error for outliers, which are defined as all data points that 
    fall above the upper_limit or below the lower_limit.
    
    Args:
    y_test (Numpy Array of shape (n,) or Pandas Series): Actual values of the response variable
    y_pred (Numpy Array of shape (n,) or Pandas Series): Predicted values of the response variable
    lower_limit (Float): Values lower than this limit are treated as outliers and used in the calculation
    upper_limit (Float): Values higher than this limit are treated as outliers and used in the calculation
        
    Returns:
    root_mean_sq_error (Float): Root mean squared error
    
    """    
    y_compare = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred})
    
    if lower_limit!=None:
        y_compare = y_compare[(y_compare['y_test']<lower_limit)|(y_compare['y_test']>upper_limit)]
        
    root_mean_sq_error = mean_squared_error(y_compare['y_test'], y_compare['y_pred'])**0.5 
    
    return root_mean_sq_error


def outliers_sensitivity(data_set_train_full, data_set_test, features_all, y_col,
                         chosen_estimator, fold_num,
                         train_limits, test_limits):
    """
    Estimates the sensitivity of model performance with respect to the definition of outliers.
    
    Args:
    data_set_train_full (Pandas Dataframe) - Dataframe containing the training set, including features and the response variable 
    data_set_test  (Pandas Dataframe) - Dataframe containing the testing set, including features and the response variable 
    features_all (List) - List of column names that contain the model's features 
    y_col (Srting) -  Name of the column containing the response variable
    chosen_estimator (Pipeline) - Pipeline chosen for sensitivity analysis 
    fold_num (Integer) - Number of folds for cross-validation within the the training set
    train_limits (List of tuples) - List containing different definitions of outliers for the training set. 
                                    Each tuple (l1, l2) means that points in the training set below the l1 percentile 
                                    and above the l2 percentile are considered outliers
    test_limits (Tuple) - Tuple defining which points in the testing set are considered typical and which are outliers. 
                          The tuple (l1, l2) indicates that points in the testing set below the l1 percentile and above 
                          the l2 percentile are classified as outliers
    
    Returns:
    errors (Pandas Dataframe) - Dataframe containing the results of the sensitivity analysis
    """
    
    estimator = chosen_estimator
    
    data_set_train_full = data_set_train_full.copy()
    y_test = data_set_test[y_col]
    
    # Determine outlier values for the test sample based on specified percentiles
    test_lower_limit = np.percentile(data_set_test[y_col], test_limits[0])
    test_upper_limit = np.percentile(data_set_test[y_col], test_limits[1])
    
    # Estimate sizes of samples
    test_size_full = len(y_test)
    # Number of 'normal' points (not outliers) in test sample
    test_normal_size = len(data_set_test[(data_set_test[y_col]<test_upper_limit)&(data_set_test[y_col]>test_lower_limit)])
    # Number of outliers in test sample
    test_size_outliers = test_size_full - test_normal_size
    
    train_size = []
    test_size = []
    cv_scores = []
    test_score = []
    test_score_not_outliers = []
    test_score_normal = []
    test_score_outliers = []
    
    # Loop through various approaches to eliminate outliers from the training sample, 
    # e.g. train_limit[1]=99 means exclusion all data points exceeding the 99th percentile 
    # train_limit[0]=1 means exclusion data points below the 1st percentile
    for train_limit in train_limits:
        #print(train_limit[1])
        
        # Determine outlier values for the train sample based on specified percentiles
        train_upper_limit = np.percentile(data_set_train_full[y_col], train_limit[1])
        train_lower_limit = np.percentile(data_set_train_full[y_col], train_limit[0])
        # Exclude outliers from the train sample
        data_set_train = data_set_train_full.copy()[(data_set_train_full[y_col]<=train_upper_limit)&\
                                                    (data_set_train_full[y_col]>=train_lower_limit)]
        data_set_train.reset_index(drop=True, inplace=True)
        X_train = data_set_train[features_all]
        y_train = data_set_train[y_col]
        
        # The size of the training sample after removing outliers
        train_size.append(len(X_train))
        
        # Model performance on a train sample using cross-validation (after removing outliers)
        panel_cv = PanelDataSplit(test_size=4, date_axis=data_set_train['Report Date'], n_splits=fold_num)
        cv_score = cross_val_score(estimator, X_train, y_train, cv=panel_cv, scoring='neg_root_mean_squared_error').mean()
        cv_scores.append(-cv_score)
        
        # Model trained on the training sample after removing outliers
        estimator.fit(X_train, y_train)
        X_test = data_set_test[features_all]
        y_predict = estimator.predict(X_test)
        
        # Model's performance on the test set, after the exclusion of data points 
        # identified as outliers in the training set
        test_err_not_outliers = estimate_root_mean_squared_error(y_test, y_predict, train_lower_limit, train_upper_limit)
        test_score_not_outliers.append(test_err_not_outliers)
        
        # The size of the training sample after removing data points identified as outliers in the training set
        test_size.append(len(data_set_test[(data_set_test[y_col]>train_lower_limit)&(data_set_test[y_col]<train_upper_limit)]))
    
        # The model's performance on the entire test dataset
        test_score.append(mean_squared_error(y_test, y_predict)**0.5)
        
        # The model's performance on the test sample for normal points 
        # (outliers are defined using test_limits)
        err_normal = estimate_root_mean_squared_error(y_test, y_predict, lower_limit=test_lower_limit, upper_limit=test_upper_limit)
        test_score_normal.append(err_normal)
        
        # The model's performance on the test sample for otliers
        # (outliers are defined using test_limits)
        err_outliers = estimate_root_mean_squared_error_outer(y_test, y_predict, 
                                                         test_lower_limit, 
                                                         test_upper_limit)
        test_score_outliers.append(err_outliers)
    
    errors = pd.DataFrame(data={
        'excluded outliers from train set': train_limits,
        'train set size (excl.outliers)': train_size,
        'train cross-validation RMSE (excl. outliers)': cv_scores,
        'test size (excl. points that were considered outliers in train)': test_size,
        'test RMSE (excl. points that were considered outliers in train)': test_score_not_outliers,
        'test RMSE for all points, samplesize='+str(test_size_full): test_score, 
        'test RMSE for typical points (between 1 and 91 percentiles in test set), samplesize='+str(test_normal_size): test_score_normal,
        'test RMSE for outliers (lowest 1% or highest 9% in test set), samplesize='+str(test_size_outliers): test_score_outliers})
    
    return errors