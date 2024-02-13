import pandas as pd
import numpy as np



def calc_changes(macro_indicators, factors_dic, shift):
    
    """
    Calculates asolute or relative changes in macro indicators values
    
    Arguments:
    macro_indicators (Pandas DataFrame): DataFrame containing macroeconomic indicator values
    factors_dic (Dictionary): Dictionary with the structure {indicator name: type of change to compute}
    shift (Integer): Number of quarters to calculate changes for

    Returns:
    macro_indicators (Pandas DataFrame): DataFrame with additional columns containing computed changes
    """
    
    macro_indicators = macro_indicators.copy()
    
    for macro_ind in factors_dic:
        
        macro_indicators[macro_ind+' change'] = \
            macro_indicators[macro_ind] - macro_indicators[macro_ind].shift(shift)
        
        if factors_dic[macro_ind] == 'relative':
            macro_indicators[macro_ind+' change'] = \
                macro_indicators[macro_ind+' change'] / macro_indicators[macro_ind].shift(shift)
            
    return macro_indicators



def generate_lagged_features(macro_indicators, factors_df):
    """
    Adds new features - lagged macro indicators
    
    Args:
    macro_indicators (Pandas Dataframe) - Dataframe containing macro indicators time series
    factors_df (Pandas DataFrame) - DataFrame containing preselected macro indicators and their relationship with the response
    
    Rerurns:
    macro_indicators (Pandas DataFrame) - Updated DataFrame with the inclusion of new lagged features
    new_factors_df (Pandas DataFrame) - DataFrame containing the list of factors
    
    """
    
    macro_indicators = macro_indicators.copy()
    
    # Dataframe with all factors, including new lagged
    new_factors_df = factors_df[['sign', 'factor group']].copy()
    
    for macro_ind in factors_df.index:
        lag = factors_df.loc[macro_ind, 'lag quarters']
        sign = factors_df.loc[macro_ind, 'sign']
        factor_group = factors_df.loc[macro_ind, 'factor group']
        if lag > 0:
            for lag_values in range(1, int(lag)+1):
                new_macro_ind = macro_ind+'_lag'+str(lag_values)
                macro_indicators[new_macro_ind] = macro_indicators[macro_ind].shift(lag_values)
                factor_row = pd.DataFrame([[sign, factor_group]], columns=['sign', 'factor group'], index=[new_macro_ind])
                new_factors_df = pd.concat([new_factors_df, factor_row])
                    
    # Dataframe with chosen macro_indicators
    columns_list = list(new_factors_df.index)
    columns_list.insert(0, "Date")
    macro_indicators = macro_indicators[columns_list]
    
    return macro_indicators, new_factors_df



def generate_ema_features(macro_indicators, features_ema, factors_df):
    """
    Adds new features - Exponential moving averages (EMA) for macro indicators
    
    Args:
    macro_indicators (Pandas Dataframe) - DataFrame with time series data of macroeconomic indicators
    features_ema (Dictionary) - Dictionary containing macro indicators for EMA transformation and 
                                the transformation parameter "center of mass" (com)
    
    Rerurns:
    macro_indicators (Pandas DataFrame) - Updated DataFrame with the inclusion of new EMA features
    new_factors_df (Pandas DataFrame) - DataFrame containing the list of factors
    """
    new_factors_df = factors_df.copy()
    macro_indicators = macro_indicators.copy()
    
    for macro_ind in features_ema:
        factor_group = new_factors_df.loc[macro_ind, 'factor group']
        sign = new_factors_df.loc[macro_ind, 'sign']
        
        # Parameter for EMA transformation
        com = features_ema[macro_ind]
        
        # New factor
        new_macro_ind = macro_ind+'_ema'+str(com)
        macro_indicators[new_macro_ind] = macro_indicators[macro_ind].ewm(com=com).mean()
        
        # Add the new factor into the dataframe listing all the factors
        factor_row = pd.DataFrame([[sign, factor_group+"_ema"]], columns=['sign', 'factor group'], index=[new_macro_ind])
        new_factors_df = pd.concat([new_factors_df, factor_row])
        
    return macro_indicators, new_factors_df