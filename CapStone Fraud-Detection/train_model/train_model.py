import pandas as pd
import numpy as np
import argparse
import sys, joblib, os, gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from azureml.core import Run

import lightgbm as lgb

run = Run.get_context()
parser = argparse.ArgumentParser("lgbm_with_hypertune")

# Input dataset
parser.add_argument("--input-data", type=str, dest='train', help='training dataset')

# Hyperparameters
parser.add_argument('--num_leaves', type=int, dest='num_leaves', default= 491, help='Number of decision leaves in a single tree')
parser.add_argument('--min_child_weight', type=float, dest='min_child_weight', default= 0.02, help='number of estimators')
parser.add_argument('--feature_fraction', type=float, dest='feature_fraction', default=0.2, help='number of estimators')
parser.add_argument('--bagging_fraction', type=float, dest='bagging_fraction', default=0.5, help='number of estimators')
parser.add_argument('--min_data_in_leaf', type=int, dest='min_data_in_leaf', default=106, help='Minimum number of data in leaf')
parser.add_argument('--max_depth', type=int, dest='max_depth', default=-1, help='Maximum depth of tree in lgbm')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.1, help='Rate of jumping in learning iterations')

# Add arguments to args collection
args = parser.parse_args()

# Log Hyperparameter values
run.log('num_leaves',  np.int(args.num_leaves))
run.log('min_child_weight',  np.float(args.min_child_weight))
run.log('feature_fraction',  np.float(args.feature_fraction))
run.log('bagging_fraction', np.float(args.bagging_fraction))
run.log('min_data_in_leaf', np.float(args.min_data_in_leaf))
run.log('max_depth', np.int(args.max_depth))
run.log('learning_rate', np.float(args.learning_rate))

train = run.input_datasets['train_data_hd']
train_pd = train.to_pandas_dataframe()

X = train_pd.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)
y = train_pd.sort_values('TransactionDT')['isFraud']
del train_pd

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 998122)

params = {'num_leaves': args.num_leaves,
          'min_child_weight': args.min_child_weight,
          'feature_fraction': args.feature_fraction,
          'bagging_fraction': args.bagging_fraction,
          'min_data_in_leaf': args.min_data_in_leaf,
          'objective': 'binary',
          'max_depth': args.max_depth,
          'learning_rate': args.learning_rate,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'n_estimators' : 100,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 998122,
         }
lgbm_classifier = lgb.LGBMClassifier(**params)
model = lgbm_classifier.fit(X_train, y_train, eval_set = [(X_val, y_val)], eval_metric = 'auc')

best_eval_score = model.best_score_
auc = best_eval_score['valid_0']['auc']
run.log('AUC', np.float(auc))

os.makedirs('outputs', exist_ok = True)
joblib.dump(value = model, filename = 'outputs/hyperdrive_lgbm_fraud.pkl')
run.complete()
















