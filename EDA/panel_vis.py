import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trim_mean


def calc_statistics(bank_panel_df, fin_variable, proportiontocut=0.05):
    """
    Estimates statistical characteristics of a financial variable: 
    trimmed mean, percentiles at 5%, 25%, 50%, 75%, and 95% levels
     
    Args:
    bank_panel_df (Pandas DataFrame) - DataFrame that holds a panel of financial variables for banks
    fin_variable (String) - Name of the column representing the financial variable of interest
    proportiontocut (Float) - Fraction used to cut off of both tails of the distribution when computing a trimmed mean
    """
    
    # Calculate statistics for the financial variable
    var_stats = bank_panel_df.groupby('Report Date')[[fin_variable]].agg([lambda x: x.quantile(0.1), 
                                                                          lambda x: x.quantile(0.25),
                                                                          lambda x: trim_mean(x, proportiontocut), 
                                                                          'median',
                                                                          lambda x: x.quantile(0.75), 
                                                                          lambda x: x.quantile(0.9)])
    var_stats.columns = var_stats.columns.droplevel(0)
    var_stats.rename(columns={'<lambda_0>': 'percentile 10%', '<lambda_1>': 'percentile 25%',
                              '<lambda_2>': 'trimmed mean',
                              '<lambda_3>': 'percentile 75%', '<lambda_4>': 'percentile 90%'}, inplace=True)
    
    return var_stats
    
    
    

def plot_response_var(bank_panel_df, fin_variable, proportiontocut=0.05):
    """
    Plots statistical characteristics of a financial variable: 
    trimmed mean, percentiles at 5%, 25%, 50%, 75%, and 95% levels
     
    Args:
    bank_panel_df (Pandas DataFrame) - DataFrame that holds a panel of financial variables for banks
    fin_variable (string) - Name of the column representing the financial variable of interest
    proportiontocut (float) - Fraction used to cut off of both tails of the distribution when computing a trimmed mean
    """
    
    # Calculate statistics for the financial variable
    var_stats = calc_statistics(bank_panel_df, fin_variable, proportiontocut)
    
    
    # Plot time series of statistics
    plt.plot(pd.to_datetime(var_stats.index), var_stats['median'], color='black', label='median')
    plt.plot(pd.to_datetime(var_stats.index), var_stats['trimmed mean'], '--', color='black', linewidth=2, 
             label='trimmed mean (cut ' + str(proportiontocut) + ')')

    plt.fill_between(pd.to_datetime(var_stats.index), var_stats['percentile 10%'], var_stats['percentile 90%'], 
                     color='red', alpha=0.2, label='10%-90% percentiles')

    plt.fill_between(pd.to_datetime(var_stats.index), var_stats['percentile 25%'], var_stats['percentile 75%'], 
                     color='red', alpha=0.5, label='25%-75% percentiles')
    
    
    # Show time intervals characterized by economic crises
    # 2008-2009 financial crisis 
    # Dates of recession are from 
    # https://www.federalreservehistory.org/essays/great-recession-of-200709#:~:text=Lasting%20from%20December%202007%20to,longest%20since%20World%20War%20II.&text=The%20Great%20Recession%20began%20in,recession%20since%20World%20War%20II.
    plt.axvspan('2007-12-31', '2009-06-30', alpha=0.2, color='grey', label='2008-2009 fin. crisis')

    # COVID recession
    # Dates of recession are from 
    # https://en.wikipedia.org/wiki/COVID-19_recession#:~:text=United%20States,-Main%20article%3A%20Economic&text=The%20National%20Bureau%20of%20Economic,on%20records%20dating%20to%201854.
    plt.axvspan('2019-12-31', '2020-03-31', alpha=0.4, color='grey', label='covid recession')

    plt.title(fin_variable, fontsize=14)
    plt.legend();  
    

    

def plot_response_independent_series(data_set, macro_indicators, 
                                     fin_variable, macro_variable,
                                     sign, shift, ema_com, proportiontocut=0.05):
    """
    """
    # Calculate statistics for the financial variable
    var_stats = calc_statistics(data_set, fin_variable)
    
    # Transfrom macro indicator
    macro_df = macro_indicators[['Date', macro_variable]].copy()
    macro_variable_transf = macro_variable + '_transf'
    # Apply sign and exponential moving average transfromation
    macro_df[macro_variable_transf] = sign * macro_df[macro_variable].ewm(com=ema_com).mean()
    # Shift values of the macro indicator
    macro_df[macro_variable_transf] = macro_df[macro_variable_transf].shift(shift)
    macro_df = macro_df[pd.to_datetime(macro_df['Date'])>=pd.to_datetime('2002-12-31')]
    
    
    fig, ax = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    
    # Plot time series of statistics
    ax[0].plot(pd.to_datetime(var_stats.index), var_stats['median'], color='black', label='median')
    ax[0].plot(pd.to_datetime(var_stats.index), var_stats['trimmed mean'], '--', color='black', linewidth=2, 
               label='trimmed mean (cut ' + str(proportiontocut) + ')')
    
    # Plot the transformed macro indicator
    ax[1].plot(pd.to_datetime(macro_df['Date']), macro_df[macro_variable_transf], color='black',
               label=macro_variable+'\nTransformation: sign='+str(sign)+', shift='+ str(shift)+', ema_com='+str(ema_com))
    ax[0].legend()
    ax[1].legend()
    plt.suptitle(' Comparison of Provision for Loan Losses with ' + macro_variable, fontsize=14, y=0.95);    
    
    
    
def plt_two_indicators(macro_indicators, indicator1, indocator2):
    """
    Generates plots for 2 time series

    Args:
    macro_indicators (Pandas DataFrame) - A DataFrame containing time series data for macroeconomic indicators
    indicator1 (String) - The name of the first indicator (column name in the macro_indicators DataFrame)
    indicator2 (String) - The name of the second indicator (column name in the macro_indicators DataFrame)
    
    """
    plt.plot(pd.to_datetime(macro_indicators['Date']), macro_indicators[indicator1], 
             label='raw indicator', color='grey', alpha=0.8)
    plt.plot(pd.to_datetime(macro_indicators['Date']), macro_indicators[indocator2], 
             label ='ema',color='black')
    plt.title(indicator1, fontsize=14)
    plt.legend();    