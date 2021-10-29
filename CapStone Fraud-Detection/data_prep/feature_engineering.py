import pandas as pd
import numpy as np
import gc
import argparse
import sys

from sklearn.preprocessing import LabelEncoder
from azureml.core import Run

############################################################################################################################################

def feature_engineering_one(dataframe : pd.DataFrame) -> pd.DataFrame:
    columns_a = ['TransactionAmt', 'id_02', 'D15']
    columns_b = ['card1', 'card4', 'addr1']

    for col_a in columns_a:
        for col_b in columns_b:
            for df in [dataframe]:
                df[f'{col_a}_to_mean_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('mean')
                df[f'{col_a}_to_std_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('std')
    
    # New feature - log of transaction amount.
    dataframe['TransactionAmt_Log'] = np.log(dataframe['TransactionAmt'])
    
    # New feature - decimal part of the transaction amount.
    dataframe['TransactionAmt_decimal'] = ((dataframe['TransactionAmt'] - dataframe['TransactionAmt'].astype(int)) * 1000).astype(int)

    # New feature - day of week in which a transaction happened.
    dataframe['Transaction_day_of_week'] = np.floor((dataframe['TransactionDT'] / (3600 * 24) - 1) % 7)

    # New feature - hour of the day in which a transaction happened.
    dataframe['Transaction_hour'] = np.floor(dataframe['TransactionDT'] / 3600) % 24

    # Some arbitrary features interaction
    for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 
                    'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:

        f1, f2 = feature.split('__')
        dataframe[feature] = dataframe[f1].astype(str) + '_' + dataframe[f2].astype(str)

        le = LabelEncoder()
        le.fit(list(dataframe[feature].astype(str).values))
        dataframe[feature] = le.transform(list(dataframe[feature].astype(str).values))

    # Encoding - count encoding for both train and test
    for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:
        dataframe[feature + '_count_full'] = dataframe[feature].map(pd.concat([dataframe[feature]], ignore_index=True).value_counts(dropna=False))

    # Encoding - count encoding separately for train and test
    for feature in ['id_01', 'id_31', 'id_33', 'id_36']:
        dataframe[feature + '_count_dist'] = dataframe[feature].map(dataframe[feature].value_counts(dropna=False))

    return dataframe

def feature_engineering_two(dataframe : pd.DataFrame) -> pd.DataFrame:

    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 
    'comcast.net': 'other','yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
    'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 
    'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 
    'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 
    'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 
    'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other','rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 
    'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 
    'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 
    'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
     
    us_emails = ['gmail', 'net', 'edu']

    for c in ['P_emaildomain', 'R_emaildomain']:
        dataframe[c + '_bin'] = dataframe[c].map(emails)
        dataframe[c + '_suffix'] = dataframe[c].map(lambda x: str(x).split('.')[-1])
        dataframe[c + '_suffix'] = dataframe[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(dataframe[col].astype(str).values))
            dataframe[col] = le.transform(list(dataframe[col].astype(str).values))

    return dataframe
    
############################################################################################################################################

print("Peform feature engineering on remaining columns")

run = Run.get_context()
selected_data = run.input_datasets["selected_data"]
selected_data_pd = selected_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("feature_engineering")
parser.add_argument("--output_train_data", type = str, help = "Final dataset for model training")

args = parser.parse_args()
print("Argument (output of final dataset for model training): %s" % args.output_train_data)

final_train_data = feature_engineering_one(selected_data_pd)
final_train_data = feature_engineering_two(final_train_data)
final_train_data.replace([np.inf, -np.inf], np.nan, inplace=True)


if not (args.output_train_data is None):
    os.makedirs(args.output_train_data, exist_ok=True)
    print("%s created" % args.output_train_data)
    path = args.output_train_data + "/train_data.parquet"
    write_df = final_train_data.to_parquet(path)












