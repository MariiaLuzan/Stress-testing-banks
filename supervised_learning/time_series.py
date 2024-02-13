import numpy as np
import pandas as pd
from itertools import combinations

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order


def create_bank_train_set(bank_id, data_set, 
                          y_col, factors_df,
                          train_up_to):
    """
    Creates data sets (whole and training) for the specified bank
    
    Args:
    bank_id (Integer) - Identifier for the bank, derived from the "IDRSSD" column
    data_set (Pandas Dataframe) - Dataframe containing tha bank panel and macroeconomic time series
    y_col (String) - Name of the column containg the response variable 
    factors_df (Pandas Dataframe) - Dataframe containing potential model features and the rules for using these factors
    train_up_to (String) - String specifying the final date for the training dataset in the format 'yyyy-mm-dd'
    
    Returns:
    bank_data (Pandas Dataframe) - Dataframe containg all rows for the specified bank, including date, 
                                   the response variable, and potential features
    bank_data_train (Pandas Dataframe) - Dataframe containing the training set for the specified bank, which includes 
                                         date, the response variable, and possible features
    
    """
    # Filter bank data
    bank_data = data_set.loc[data_set['IDRSSD']==bank_id, ['Report Date', y_col]+list(factors_df.index)]
    bank_data.reset_index(drop=True, inplace=True)
    
    # Create train and test samples
    bank_data_train = bank_data[bank_data["Report Date"] <= train_up_to]
        
    return bank_data, bank_data_train



def find_EMA_com(com_vals, bank_data_train_wt_nans, macro_ind, sign):
    """
    Determines the optimal Exponential Moving Average (EMA) parameter to maximize 
    the correlation between AR residuals and the values of a specific macroeconomic indicator
    
    Args:
    com_vals (List) - List of potential EMA parameter values from which to select the optimal value
    bank_data_train_wt_nans (Pandas Dataframe) - Dataframe containing the bank's training dataset, excluding NaN values
    macro_ind (String) - Name of the column containing the values of the macroeconomic indicator
    sign (Integer) - Sign of the relationship between residuals and the specified macroeconomic indicator. 
                     Can take one of two values: 1 or -1
    
    Returns:
    optim_com (Float) - Optimal value for the EMA's parameter
    corr_opt (Float) - correlation coefficient between residuals and macroeconomic indicator values, 
                       transformed using EMA with the optimal parameter
    """
    corr_coefs = []
    for com_param in com_vals:
        corr_coef = np.corrcoef(bank_data_train_wt_nans['residuals'], 
                                bank_data_train_wt_nans[macro_ind].ewm(com=com_param).mean())[0,1]
        corr_coefs.append(corr_coef)
    
    corr_coefs = np.array(corr_coefs)

    if sign == -1:
        com_indx = corr_coefs.argmin()
        corr_opt = corr_coefs.min()
    else:
        com_indx = corr_coefs.argmax()
        corr_opt = corr_coefs.max()
    
    optim_com = com_vals[com_indx] 
    
    return optim_com, corr_opt



def calc_corr_factors_residuals(factors_df, bank_data_train, com_vals):
    """
    Calculates the correlation between residuals from an AutoRegressive (AR) model and 
    macroeconomic indicators for a single bank. It filters out factors with an incorrect 
    sign with the residuals.
    
    Args:
    factors_df (Pandas Dataframe) - Dataframe containing potential model features and the rules for using these factors
    bank_data_train (Pandas Dataframe) - Dataframe containg the training set for one bank, including date, 
                                         the response variable, and potential features 
    com_vals (List) - List of potential EMA parameter values from which to select the optimal value
    
    Returns:
    factors_bank (Pandas Dataframe) - DataFrame containing correlation coefficients between residuals 
                                      and macroeconomic indicators for one bank
    """
    
    bank_data_train_wt_nans = bank_data_train.dropna()
    
    # Prepare dataframe for estimated correlations
    factors_bank = factors_df.copy()
    factors_bank['corr'] = np.NaN
    # Parameter for EMA
    factors_bank['com'] = np.NaN
    factors_bank['exclude'] = 0
    
    for macro_ind in factors_bank.index:
        sign = factors_bank.loc[macro_ind, 'sign']
        calc_ema = factors_bank.loc[macro_ind, 'calc_ema']
    
        if calc_ema == 1: 
            optim_com, corr_coef = find_EMA_com(com_vals, bank_data_train_wt_nans, macro_ind, sign)
            factors_bank.loc[macro_ind, 'com'] = optim_com
        else:
            corr_coef = np.corrcoef(bank_data_train_wt_nans['residuals'],
                                    bank_data_train_wt_nans[macro_ind])[0, 1]
    
        factors_bank.loc[macro_ind, 'corr'] = corr_coef
        # Check that sigh of correlation coefficicent is correct
        if sign * corr_coef < 0:
            factors_bank.loc[macro_ind, 'exclude'] = 1     

    factors_bank = factors_bank[factors_bank['exclude']==0] 
    factors_bank['corr_abs'] = np.abs(factors_bank['corr'])
    
    return factors_bank


def exclude_factors(factors_bank, corr_limit):
    """
    Excludes factors from the "factors_bank" DataFrame that have a weak correlation 
    with the residuals of the AutoRegressive (AR) model
    
    Args:
    factors_bank (Pandas Dataframe) - DataFrame containing correlation coefficients between residuals 
                                      and macroeconomic indicators for one bank 
    corr_limit (Float) - Factors with correlation coefficients below the specified "corr_limit"
                         will be removed from the "factors_bank" DataFrame
    
    Returns:
    factors_bank  (Pandas Dataframe) - DataFrame containing correlation coefficients between residuals and 
                                       macroeconomic indicators for one bank after excluding factors with 
                                       correlations below the threshold
    """
    
    # Exclude factors with the wrong sign
    factors_bank = factors_bank[factors_bank['exclude']!=1]
    
    # In a group of factors choose factor with highest correlation
    factors_bank['max_corr_in_group'] = factors_bank.groupby('group')[['corr_abs']].transform("max")
    factors_bank['choose_in_group'] = np.where(factors_bank['corr_abs']==factors_bank['max_corr_in_group'], 1, 0)
    
    corr_limit = min(corr_limit, factors_bank['corr_abs'].max())
    factors_bank = factors_bank[(factors_bank['choose_in_group']==1)&(factors_bank['corr_abs']>=corr_limit)]
    
    return factors_bank


def apply_ema(factors_bank, bank_data):
    """
    Applies Exponential Moving Average (EMA) transformation on selected factors
    
    Args:
    factors_bank (Pandas Dataframe) - DataFrame containing correlation coefficients between residuals and 
                                      macroeconomic indicators, as well as optimized EMA parameters for one bank 
    bank_data (Pandas Dataframe) - Dataframe containg all rows for the specified bank, including date, 
                                   the response variable, and potential features
                                   
    Returns:
    bank_data (Pandas Dataframe) - Updated dataframe with factors transformed using EMA
    """
    
    # Factors for which EMA should be applied
    factors_ema = factors_bank[factors_bank['calc_ema']==1].index
    
    for indicator in factors_ema:
        com_param = factors_bank.loc[indicator, 'com']
        bank_data[indicator] = bank_data[indicator].ewm(com=com_param).mean()
        
    return bank_data 


def check_model_signs(params, factors_bank):
    """
    Verifies the correctness of the signs of the model's coefficients
    
    Args:
    params (Pandas Series) - Series containing the estimated coefficients of the model
    factors_bank (Pandas Dataframe) - Dataframe that includes the names of factors and their correct signs
    
    
    Returns:
    excl_model (Integer) - Returns 1 if all signs of factors in the model are correct, and 0 otherwise
    """
    
    # Model coefficients
    coef_df = pd.DataFrame(params)
    coef_df.rename(columns={0: 'coef'}, inplace=True)
    
    # Compare signs of model coefficients with correct signs from the factors_bank dataframe
    coef_df = coef_df.merge(factors_bank['sign'], how='left', left_index=True, right_index=True)
    coef_df['correct_sign'] = np.where(coef_df['sign']*coef_df['coef'] > 0, 1, 0)
    coef_df['correct_sign'] = np.where(coef_df['sign'].isnull(), np.NaN, coef_df['correct_sign'] )
    coef_df['exclude_model'] = np.where(coef_df['correct_sign']==0, 1, 0)
    excl_model = min(coef_df['exclude_model'].sum(), 1)
    
    return excl_model  



def model_combinations(y_col, factors_bank, bank_data_train, autocor_order):
    """
    Generates models for a single bank by considering all possible combinations of factors 
    and selects the best model based on the Akaike Information Criterion (AIC).
    
    Args:
    y_col (String) - Name of the column containg the response variable  
    factors_bank (Pandas Dataframe) - DataFrame containing correlation coefficients between residuals and 
                                      macroeconomic indicators
    bank_data_train (Pandas Dataframe) - Dataframe containing the training set for the specified bank, which includes 
                                         date, the response variable, and possible features
    autocor_order (Integer) - order of the autocorrelation time series model
    
    Returns:
    models_info (Pandas Dataframe) - DataFrame that presents results for all potential models, 
                                     including correctness of signs and AIC scores
    best_model (List) - List of the macroeconomic indicators from the best model
    best_params (Pandas Series) - Estimated coefficients for the best model
    
    """

    factors_list = list(factors_bank.index)
    
    # AIC results for models
    models_res = []
    # Models (as sets of factors)
    models = []
    # Flag: exclude model if it has coefficients with wrong signs
    excl_models = []
    # Model params
    param_models = []

    for factors_num in range(1, len(factors_list)+1):
        for comb in combinations(factors_list, factors_num):
            model = list(comb)
            models.append(model)
                    
            AR_exog_model = AutoReg(bank_data_train[y_col], exog=bank_data_train[model],
                                    lags=autocor_order).fit()
            
            param_models.append(AR_exog_model.params)
    
            # Check signs of the model coefficients
            excl_model = check_model_signs(AR_exog_model.params, factors_bank)
            excl_models.append(excl_model)
    
            # Criterium to choose the best model AIC
            model_res = AR_exog_model.aic
            models_res.append(model_res)
            
    models_info = pd.DataFrame(data={'model': models, 'wrong_sign': excl_models, 'AIC': models_res})  
    best_model = models_info.iloc[models_info['AIC'].argmin()][0]
    best_params = param_models[models_info['AIC'].argmin()]
    best_params = pd.DataFrame(best_params)
    best_params.rename(columns={0: 'coefficient'}, inplace=True)
    
    return models_info, best_model,best_params


def choose_bank_best_model(bank_id, bank_data_train, 
                           y_col, factors_df, 
                           com_vals, corr_limit=0.2):
    """
    Function that contains all the stages of constructing a model for a single bank
    
    Args:
    bank_id (Integer) - Identifier for the bank, derived from the "IDRSSD" column 
    bank_data_train (Pandas Dataframe) - Dataframe containing the training set for the specified bank, which includes 
                                         date, the response variable, and possible features 
    y_col (String) - Name of the column containg the response variable
    factors_df (Pandas Dataframe) - Dataframe containing potential model features and the rules for using these factors 
    com_vals (List) - List of potential EMA parameter values from which to select the optimal value
    corr_limit (Float) - Factors with correlation coefficients below the specified "corr_limit"
                         will be removed from the "factors_bank" DataFrame
    
    Returns:
    models_info (Pandas Dataframe) - DataFrame that presents results for all potential models, 
                                     including correctness of signs and AIC scores
    best_model (List) - List of the macroeconomic indicators from the best model 
    best_params (Pandas Series) - Estimated coefficients for the best model
    factors_bank (Pandas Dataframe) - DataFrame containing correlation coefficients between residuals and 
                                      macroeconomic indicators, and also estimated EMA's parameters
    
    """
    bank_data_train = bank_data_train.copy()
    
    autocor_order=1
    
    # Train AR model and find residuals
    AR_model = AutoReg(bank_data_train[y_col], lags=autocor_order).fit()
    bank_data_train['residuals'] = AR_model.resid
    
    # Find correlations of the model residuals with macroeconomic indicators
    factors_bank = calc_corr_factors_residuals(factors_df, bank_data_train, com_vals)
    
    # Exclude factors with low correlation
    factors_bank = exclude_factors(factors_bank, corr_limit)
    
    # Transform factors for which EMA transformation is required
    bank_data_train = apply_ema(factors_bank, bank_data_train)
    
    # Check all models - all possible factors combinations
    models_info, best_model, best_params = model_combinations(y_col, factors_bank, bank_data_train, autocor_order)
    
    return models_info, best_model, best_params, factors_bank   


########## Functions for Prediction
# Add lags of the response into bank_data
def create_response_lags(best_params, bank_data, y_col):
    """
    Adds lagged values of the response variable into the 'bank_data' dataframe
    
    Args:
    best_params (Pandas Dataframe) - Dataframe containing coefficients of the best model for the bank 
    bank_data (Pandas Dataframe) - Dataframe containg all rows for the specified bank, including date, 
                                   the response variable, and potential features
    y_col (String) - Name of the column containg the response variable 
    
    Returns:
    bank_data (Pandas Dataframe) - Updated dataframe with lagged versions of the response variable included
    """
    # Find all lagged response variables in the model features
    response_lags = [var for var in list(best_params.index) if y_col in var]
    for var in response_lags:
        # Take the lag from the named of lagged variable
        lag = var[len(y_col)+2:]
        bank_data[var] = bank_data[y_col].shift(1)
    return bank_data



def predict_response_1_date(date, best_params, bank_data):
    """
    Predicts the resonse variable value for one reporting date ahead
    
    Args:
    date (Pandas datetime) - Date for which the prediction is made
    best_params (Pandas Dataframe) - Dataframe containing coefficients of the best model for the bank 
    bank_data (Pandas Dataframe) - Dataframe containg all rows for the specified bank, including date, 
                                   the response variable, potential features,  and lagged values of 
                                   the response variable
    
    Returns:
    y_pred (Float) - Predicted value of the response variable
    """
    
    for factor in best_params.index:
        if factor=='const':
            best_params.loc[factor, 'vals'] = 1
        else:
            best_params.loc[factor, 'vals'] = bank_data.loc[date, factor]
        
    best_params['coef*vals'] = best_params['coefficient'] * best_params['vals']  
    y_pred = best_params['coef*vals'].sum()
    return y_pred


# Prediction for a bank:
def bank_prediction(bank_id, bank_data,  # bank info
                    best_params, factors_bank, # model info
                    data_set, y_col, train_up_to):
    """
    Predicts the response variable values for all dates within the test set
    
    Args:
    bank_id (Integer) - Identifier for the bank, derived from the "IDRSSD" column 
    bank_data (Pandas Dataframe) - Dataframe containg all rows for the specified bank, including date, 
                                   the response variable, and potential features,  
    best_params (Pandas Dataframe) - Dataframe containing coefficients of the best model for the bank  
    factors_bank (Pandas Dataframe) - DataFrame containing correlation coefficients between residuals and 
                                      macroeconomic indicators, and also estimated EMA's parameters
    data_set (Pandas Dataframe) - Dataframe containing tha bank panel and macroeconomic time series (for all banks)
    y_col (String) - Name of the column containg the response variable 
    train_up_to (String) - String specifying the final date for the training dataset in the format 'yyyy-mm-dd'
    
    """
    
    ### Prepare data for a prediction task
    # Transform EMA factors 
    best_params = best_params.merge(factors_bank[['calc_ema', 'com']], how='left', left_index=True, right_index=True)
    bank_data = apply_ema(best_params, bank_data)
    
    # Add lags of the response into bank_data
    bank_data = create_response_lags(best_params, bank_data, y_col)
    
    ### Prediction task
    bank_data.set_index('Report Date', inplace=True)
    # Prediction dates
    prediction_dates = bank_data.index[bank_data.index > train_up_to]
    
    for date in prediction_dates:
        y_pred = predict_response_1_date(date, best_params, bank_data)
        #print(date, y_pred)
        data_set.loc[(data_set['Report Date']==date)&(data_set['IDRSSD']==bank_id), 'y_pred'] = y_pred