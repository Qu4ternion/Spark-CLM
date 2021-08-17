# -*- coding: utf-8 -*-
"""
/!\ THIS CODE SHOULD BE EXECUTED ON THE PYSPARK SHELL USING:
        "./bin/spark-submit spark-tuning.py"

Instead of using the normal Grid Search in "xgboost.py", you can use this one 
to distribute the search across a multi-node cluster for faster convergence.

Initiate a Spark Standalone cluster using the scripts:
        "./bin/start-master.sh"
        "./bin/start-worker.sh <MASTER_URL>"
"""

from sklearn.utils import parallel_backend
from sklearn.model_selection import GridSearchCV 
from joblibspark import register_spark
import pandas as pd
import xgboost as xgb

# register spark backend
register_spark()

# Data:
y = r'C:\Users\Acer\Desktop\CLM project\data\y_test.xlsx'
X = r'C:\Users\Acer\Desktop\CLM project\data\X_test.xlsx'

X_test = pd.read_excel(X)
y_test = pd.read_excel(y)

# Grid space:
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [1, 3, 5]
    }

clf = xgb.XGBClassifier(objective='binary:logistic', missing = 1)

grid = GridSearchCV(
                estimator = xgb.XGBClassifier(objective='binary:classifier',
                                              subsample=0.9,
                                              colsample_bytree = 0.5),
                param_grid = param_grid,
                scoring = 'roc_auc',
                verbose = 0,
                n_jobs = 3,
                cv=3
                )

with parallel_backend('spark', n_jobs=3):
  grid.fit(X_test, y_test, cv=5)