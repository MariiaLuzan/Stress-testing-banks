import pandas as pd
import numpy as np
from zipfile import ZipFile
import glob

def load_call_reports(call_reports_folder, files_dic):
    """
    Loads panel bank data from call reports stored in zipped folders into a dataframe.
    
    The call reports should be retrieved from the URL: https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx 
    by selecting the "Call Reports -- Single period" option and specifying the desired Reporting Period End Date
    
    Each zipped folder contains call reports for a specific report date.
    This function reads files from the zipped folders and compiles them into a single dataframe.

    Args:
    call_reports_folder (str): Path to the folder containing zipped folders with call reports.
    files_dic (dict): Dictionary specifying the desired call reports to load and the columns to extract from them.

    Returns:
    df_call_reports (pd.DataFrame): A dataframe containing the concatenated data from the specified call reports.
    
    """
    
    # Define a dataframe that will contain concatenated data
    df_call_reports = None

    # Iterate through each zipped folder in the call_reports_folder
    for zip_file in glob.iglob(call_reports_folder+'/*'):
        
        # Take the report date (from the name of a zipped folder)
        report_date = zip_file[-12:-4]
        
        # List of call reports in the zipped folder
        call_reports = ZipFile(zip_file).namelist()
    
        # Define a dataframe to hold data for a single report date
        df_report_date = None
    
        # Find call reports that we need to load
        for files_dic_key in files_dic.keys():
        
            # Skip the row containing financial metric names
            if files_dic_key=='FFIEC CDR Call Bulk POR':
                skiprows=None
            else:
                skiprows=0
            
            for call_report in call_reports:
                call_report_name_load = files_dic_key + " " + report_date + ".txt"
                if call_report_name_load==call_report:
                    df = pd.read_csv(ZipFile(zip_file).open(call_report), sep='\t', 
                                     low_memory=False, header=0, 
                                     skiprows=skiprows)
                
                    # Financial metric codes can change over time; for instance, in the earlier periods, 
                    # the provision for loan losses had the code 'RIAD4230,' later changing to 'RIADJJ33'.
                    # Consequently, various report dates may have varying columns for certain financial metrics. 
                    # We should only load columns that currently exist in the dataframe.
                    set_df = set(df.columns)
                    set_dic = set(files_dic[files_dic_key].keys())
                    columns = list(set.intersection(set_dic, set_df))
                               
                    df = df[columns]
                
                    if type(df_report_date)==pd.core.frame.DataFrame:
                        df_report_date = df_report_date.merge(df, on="IDRSSD")
                    else:
                        df_report_date = df.copy()
                
                    break
    
        # Add report date as a column
        df_report_date['Report Date'] = report_date
    
        # Concatenate dataframes for different report dates    
        #print(type(df_call_reports))
        if type(df_call_reports)==pd.core.frame.DataFrame:
            df_call_reports = pd.concat([df_call_reports, df_report_date])
        else:
            df_call_reports = df_report_date.copy() 

    # Specify the order of the columns
    df_call_reports = df_call_reports.reindex(columns=(['Report Date'] + list([col for col in df_call_reports.columns if col!='Report Date'])))
    
    df_call_reports['Report Date'] = pd.to_datetime(df_call_reports['Report Date'], format='%m%d%Y')
    df_call_reports.sort_values(by=['Report Date', 'IDRSSD'], inplace=True)
    
    df_call_reports.reset_index(drop=True, inplace=True)
    
    return df_call_reports
