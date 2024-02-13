import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



def estimate_r2(y_test, y_pred, lower_limit = None, upper_limit=None):
    """
    Calculate R-squared statistic excluding outliers, which are defined as all data points that 
    fall above the upper_limit or below the lower_limit.
    
    Args:
    y_test (Numpy Array of shape (n,) or Pandas Series): Actual values of the response variable
    y_pred (Numpy Array of shape (n,) or Pandas Series): Predicted values of the response variable
    lower_limit (Float): Values lower than this limit are treated as outliers and excluded from the calculation
    upper_limit (Float): Values higher than this limit are treated as outliers and excluded from the calculation
        
    Returns:
    r_squared (Float): R-squared statistic
    """
    
    y_compare = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred})
    
    if lower_limit!=None:
        y_compare = y_compare[(y_compare['y_test']>lower_limit)&(y_compare['y_test']<upper_limit)]
    
    r_squared = r2_score(y_compare['y_test'], y_compare['y_pred'])
    
    return r_squared


def estimate_median_relative_error(y_test, y_pred, lower_limit = None, upper_limit=None):
    """
    Estimates the median relative error excluding outliers, which are defined as all data points that 
    fall above the upper_limit or below the lower_limit.
    To prevent division by zero, the function replaces any zero values in y_test with a very small number.

    Args:
    y_pred (Numpy Array of shape (n,) or Pandas Series): Predicted values of the response variable
    y_test (Numpy Array of shape (n,) or Pandas Series): Actual values of the response variable
    lower_limit (Float): Values lower than this limit are treated as outliers and excluded from the calculation
    upper_limit (Float): Values higher than this limit are treated as outliers and excluded from the calculation
        
    Returns:
    median_relative_error (Float): Median relative error
    
    """
    y_compare = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred})
    
    if lower_limit!=None:
        y_compare = y_compare[(y_compare['y_test']>lower_limit)&(y_compare['y_test']<upper_limit)]
    
    median_relative_error = np.median(np.abs(y_compare['y_test'] - y_compare['y_pred']) / \
                                      np.where(y_compare['y_test']==0, 0.0000001, y_compare['y_test'])) * 100
    
    return median_relative_error  


def estimate_mean_relative_error(y_test, y_pred, lower_limit = None, upper_limit=None):
    """
    Calculates the mean relative error excluding outliers, which are defined as all data points that 
    fall above the upper_limit or below the lower_limit.
    To avoid division by zero, the function excludes data points where y_test is equal to zero. 
    Additionally, the function provides the average y_pred for cases when y_test equals zero.

    Args:
    y_test (Numpy Array of shape (n,) or Pandas Series): Actual values of the response variable
    y_pred (Numpy Array of shape (n,) or Pandas Series): Predicted values of the response variable
    lower_limit (Float): Values lower than this limit are treated as outliers and excluded from the calculation
    upper_limit (Float): Values higher than this limit are treated as outliers and excluded from the calculation
    
    Returns:
    mean_relative_error (Float): Mean relative error, computed while excluding cases where y_test is equal to zero
    y_pred_mean_zeros (Float): Average y_pred for cases when y_test equals zero
    
    """
    
    y_compare = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred})
    
    if lower_limit!=None:
        y_compare = y_compare[(y_compare['y_test']>lower_limit)&(y_compare['y_test']<upper_limit)]
        
    
    y_compare_wt_zeros = y_compare[y_compare['y_test']!=0]
    
    mean_relative_error = \
        (np.abs((y_compare_wt_zeros['y_pred'] - y_compare_wt_zeros['y_test']) / y_compare_wt_zeros['y_test'])).mean() * 100
    
    y_pred_mean_zeros = y_compare.loc[y_compare['y_test']==0, 'y_pred'].mean()
    
    return mean_relative_error, y_pred_mean_zeros   


def estimate_errors(y_test, y_pred, lower_limit = None, upper_limit=None, calc_r2=True):
    """
    Calculates R squared, RMSE, and median relative error excluding outliers, 
    which are defined as all data points that fall above the upper_limit or below the lower_limit.
    
    Args:
    y_pred (Numpy Array of shape (n,) or Pandas Series): Predicted values of the response variable
    y_test (Numpy Array of shape (n,) or Pandas Series): Actual values of the response variable
    lower_limit (Float): Values lower than this limit are treated as outliers and excluded from the calculation
    upper_limit (Float): Values higher than this limit are treated as outliers and excluded from the calculation
    
    Returns:
    errors_df (Pandas DataFrame): Dataframe containing estimated values    
    """
    if calc_r2==True:
        errors_df = pd.DataFrame(index=['R squared', 'RMSE', 'median relative error, %'],
                                 columns=['measure'])
    else:
        errors_df = pd.DataFrame(index=['RMSE', 'median relative error, %'],
                                 columns=['measure'])
    
    y_compare = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred})
    
    if lower_limit!=None:
        y_compare = y_compare[(y_compare['y_test']>lower_limit)&(y_compare['y_test']<upper_limit)]
    
    if calc_r2==True:
        errors_df.loc['R squared', 'measure'] = estimate_r2(y_test, y_pred, lower_limit, upper_limit)
    errors_df.loc['RMSE', 'measure'] = mean_squared_error(y_compare['y_test'], y_compare['y_pred'])**0.5
    errors_df.loc['median relative error, %', 'measure'] = estimate_median_relative_error(y_test, y_pred, lower_limit, upper_limit)
    
    
                     
    return errors_df