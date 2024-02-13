import pandas as pd
import numpy as np


def merge_bank_macro_datasets(bank_data, macro_data, pca_data, macro_data1):
    
    """
    Combines two data sources: Bank panel data and macroeconomic indicators.
    Prepares the sample: 
    - creates lagged response variable
    - excludes data for the year 2023 where macroeconomic data is unavailable
    
    Args:
    bank_data (Pandas Dataframe) - Dataframe containing bank panel data 
    macro_data (Pandas Dataframe) - Dataframe containing time series of macroeconomic indicators
    pca_data (Pandas Dataframe) - Dataframe containing PCA of time series of macroeconomic indicators
    macro_data1 (Pandas Dataframe) - Dataframe containing additional macroeconomic indicators
    
    Returns:
    merged_df (Pandas Dataframe) - DataFrame that integrates the bank panel data and macroeconomic indicators
    """
    
    # Convert 'Report Date' and 'Date' to datetime objects
    bank_data['Report Date'] = pd.to_datetime(bank_data['Report Date'])
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])
    pca_data['Date'] = pd.to_datetime(pca_data['Date'])
    macro_data1['Date'] = pd.to_datetime(macro_data1['Date'])

    # Merge DataFrames on 'Report Date' and 'Date'
    merged_df = pd.merge(bank_data[['Report Date', 'IDRSSD',
                                    'Financial Institution Name',
                                    'Provision for Loan Lease Losses as % of Aver. Assets']], 
                         macro_data, left_on='Report Date', right_on='Date', how='left')
    
    merged_df = merged_df.merge(pca_data, left_on='Report Date', right_on='Date', how='left')
    merged_df = merged_df.merge(macro_data1, left_on='Report Date', right_on='Date', how='left', suffixes=('', '_dupl'))
    
    # Drop duplicate columns
    merged_df.drop(columns=[col for col in merged_df.columns if '_dupl' in col], inplace=True)

    # Drop the duplicate 'Date' column
    merged_df = merged_df.drop(columns=['Date', 'Date_x'])
    
    # Create lagged response variable
    merged_df['Provision_Lag1'] = merged_df.groupby('IDRSSD')['Provision for Loan Lease Losses as % of Aver. Assets'].shift(1)
    
    # Remove the data for the year 2023 as there is no available macroeconomic data for that year 
    merged_df = merged_df[merged_df['Report Date']<='2022-12-31']
    
    return merged_df