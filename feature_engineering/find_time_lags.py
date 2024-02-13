import pandas as pd
import numpy as np
from EDA.panel_vis import calc_statistics



def calc_lags(response_var, macro_indicators,
              fin_variable, factors_df, 
              start_date, end_date):
    
    """
    Calculates time lags between the extremes of the response variable and macroeconomic indicators
    
    Args:
    response_var (Pandas DataFrame): DataFrame containing a panel with various response variable variants
    macro_indicators (Pandas DataFrame): DataFrame with macroeconomic time series data
    fin_variable (String): The selected response variable (column name in the response_var dataframe)
    factors_df (Pandas DataFrame): DataFrame containing preselected macro indicators and their relationship with the response
    start_date, end_date (String): The start and end dates defining the period for calculating the maximums, 
                                  in the 'YYYY-MM-DD' format.

    Returns:
    factors_df (Pandas DataFrame): DataFrame with an additional column indicating the time lag in quarters
    
    """
    
    ### 1. Find the date of the maximal trimmed mean for the response variable
    # Generate a table with statistics for the response variable, including the trimmed mean 
    response_var_stats = calc_statistics(response_var, fin_variable)
    response_var_stats.index = pd.to_datetime(response_var_stats.index)
    # Restrict the statistics dataframe to encompass only the period between the start_date and end_date
    response_var_stats = response_var_stats[(response_var_stats.index>=start_date)&(response_var_stats.index<=end_date)]
    # The date of the maximal trimmed mean 
    response_extremum_date = response_var_stats.index[response_var_stats['trimmed mean'].argmax()]
    
    
    ### 2. Transform the macro_indicators dataframe for convenience
    macro_indicators = macro_indicators.copy()
    macro_indicators['Date'] = pd.to_datetime(macro_indicators['Date'])
    macro_indicators.set_index('Date', inplace=True)
    # Restrict the macro_indicators dataframe to encompass only the period between the start_date and end_date
    macro_indicators_per = macro_indicators[(macro_indicators.index>=start_date)&(macro_indicators.index<=end_date)]
    
    
    ### 3. Find the date of the extremum for each macro indicator listed in the factors_df table
    factors_df = factors_df.copy()
    factors_df['lag quarters'] = np.NaN
    factors_df['factor group'] = np.NaN
        
    for macro_ind in factors_df.index:
        sign = factors_df.loc[macro_ind, 'sign']
        # The date of the maximal value for the macro indicator
        macro_extremum_date = macro_indicators_per.index[(sign*macro_indicators_per[macro_ind]).argmax()]
        # Find the time lag between the extremum of the response variable and that of the macroeconomic indicator
        lag = response_extremum_date.to_period('M') - macro_extremum_date.to_period('M')
        # Convert the lag into the number of quarters
        lag_months = (str(lag)[:-13])[1:]
        lag_months = int(lag_months)
        factors_df.loc[macro_ind, 'lag quarters'] = lag_months / 3
        # Add "group of factors" column
        factors_df.loc[macro_ind, 'factor group'] = macro_ind
        
        
        
    return factors_df   



