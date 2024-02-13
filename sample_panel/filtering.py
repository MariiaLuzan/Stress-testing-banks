import pandas as pd
import numpy as np


def bank_filtering(df):
    
    # find banks whose total assets are more than 10 billion in at least one quarter from 2001 to 2023/6/30
    df1 = df[df['UBPR2170'] > 10000000]
    # find banks whose total deposits are more than 1 million in at least one quarter from 2001 to 2023/6/30
    df2 = df1[df1['UBPR2200'] >= 1000]
    # the bank list 
    banks = df2['IDRSSD'].unique().tolist()
    
    bankfiltered_df = df[df['IDRSSD'].isin(banks)]
    
    
    bankfiltered_df = bankfiltered_df[bankfiltered_df['Report Date']>='2002-12-31']
    
    bankfiltered_df = bankfiltered_df.reset_index(drop=True)
    
    return bankfiltered_df