import pandas as pd
import numpy as np


def response_variables(df):
    
    #'UBPRPG64': 'Pre Provision Net Revenue YTD $ (TE)' 
    #'UBPRD659': 'Average Total Assets ($000)'
    df['Quarter'] = pd.to_datetime(df['Report Date']).dt.month / 3
    df['Pre Provision Net Revenue as % of Aver. Assets'] = 4 * df['UBPRPG64'] / df['Quarter'] / df['UBPRD659'] * 100
    
    
    # Provision per quarter
    df['Year'] = pd.to_datetime(df['Report Date']).dt.year
    df['Month'] = pd.to_datetime(df['Report Date']).dt.month
    # 'RIAD4230' and 'RIADJJ33' are provisions from call reports
    df['Provision'] = np.where(df['RIAD4230'].isnull(),
                               df['RIADJJ33'],
                               df['RIAD4230'])
    df['Provision_lag1'] = df.groupby(['IDRSSD', 'Year'])['Provision'].shift(1)
    df['Provision_quarter'] = np.where(df['Month']==3,
                                       df['Provision'],
                                       df['Provision'] - df['Provision_lag1'])
    # 'UBPR2170' - Total assets from risk reports
    df['Provision_quarter as % of Assets'] = 4 * df['Provision_quarter'] / df['UBPR2170'] * 100
    
    
    df_filt = df[['Report Date', 'IDRSSD', 'Financial Institution Name', 
                  'UBPRE006', 'Provision_quarter as % of Assets',
                  'UBPRE003', 'UBPRE004', 
                  'Pre Provision Net Revenue as % of Aver. Assets', 'UBPRD486', 'UBPRD488', 
                  'UBPRE591', 'UBPRE598', 'UBPRE599', 'UBPRE600',
                  'RIAD4230', 'RIADJJ33', 'UBPR2170', 'UBPRD659']].copy()
    
    df_filt.rename(columns={'UBPRE003': 'Net Interest Income as % of Aver. Assets',
                            'UBPRE004': 'Noninterest Income as % of Aver. Assets',
                            'UBPRE006': 'Provision for Loan Lease Losses as % of Aver. Assets', 
                            'UBPRE591': 'Core Deposits as % of Total Assets',
                            'UBPRE598': 'Short Term Assets as % of Short Term Liabilities', 
                            'UBPRE599': 'Net Short Term Liabilities as % of Total Assets', 
                            'UBPRE600': 'Net Loans Leases as % of Total Deposits',
                            'UBPRD486': 'Tier One Leverage Capital',
                            'UBPRD488': 'Total Risk-Based Capital to Risk-Weighted Assets',
                            }, 
                   inplace=True)
    
    
    return df_filt