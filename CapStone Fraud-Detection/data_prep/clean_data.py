import argparse
import os
import pandas as pd
import numpy as np
import gc

from azureml.core import Run
from sklearn.preprocessing import LabelEncoder

############################################################################################################################################

def id_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data and split some useful data from each columns for identity dataset.
    Args:
        - dataframe : pd.DataFrame
    Output:
        - dataframe : pd.DataFrame
    """
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1    
    return dataframe

############################################################################################################################################

print("Clean customer identity and transaction data before combine together")

run = Run.get_context()

parser = argparse.ArgumentParser("clean-data")
parser.add_argument("--output_combine", type = str, help = "output of cleaned identity data and transaction data as unified table")

args = parser.parse_args()

print("Argument 1(output cleaned data of identity data and transaction data): %s" % str(args.output_combine))

raw_transaction = run.input_datasets['input_transaction']
raw_identity = run.input_datasets['input_identity']

raw_transaction_pd = raw_transaction.to_pandas_dataframe().set_index(['TransactionID'])
raw_identity_pd = raw_identity.to_pandas_dataframe().set_index(['TransactionID'])

raw_identity_pd = id_split(raw_identity_pd)
train = raw_transaction_pd.merge(raw_identity_pd, how='left', left_index=True, right_index=True)

if not (args.output_combine is None):
    os.makedirs(args.output_combine, exist_ok=True)
    print("%s created" % args.output_combine)
    path = args.output_combine + "/cleaned_combine.parquet"
    write_df = train.to_parquet(path)











