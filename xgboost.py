# -*- coding: utf-8 -*-

import xgboost as xgb
from pyspark.sql import SparkSession

from sklearn.model_selection  import train_test_split, GridSearchCV
from sklearn.metrics          import (plot_confusion_matrix, precision_score,
                                      accuracy_score, f1_score)
import pandas as pd
from joblib import parallel_backend

# Start a Spark Session:

spark = SparkSession\
        .builder\
        .appName("PySpark XGBOOST Titanic")\
        .master("local[*]")\
        .getOrCreate()

# Import data:        
path = r'C:\Users\Acer\Desktop\CLM project\data\CLM.xlsx'
dt = pd.read_excel(path)

# Drop non-needed columns:
dt = dt.drop(['Unnamed: 0', 'index', 'client_id', 'transaction_date',
              'subscription_date', 'churn_date'], 1)

# The target classes are pretty balanced: almost 50-50
dt['churned'].groupby(by=dt.churned).count() / len(dt)

# Independent variables:
X = dt.drop(columns = 'churned')

# One-hot encoding of nominal variable:
X = pd.get_dummies(X, columns = ['product_bought',
                                 'subscription_type'])
# Target variable:
y = dt['churned']

# Type safety:
X.dtypes
y.dtypes

# Splitting data:
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Export testing data for Spark-tuning:
X_test.to_excel(r'C:\Users\Acer\Desktop\CLM project\data\X_train.xlsx')
y_test.to_excel(r'C:\Users\Acer\Desktop\CLM project\data\y_train.xlsx')

# XGBoost:
clf = xgb.XGBClassifier(objective='binary:logistic', missing = 1)

# Fit the model:
clf.fit(X_train,
        y_train,
        verbose = True,
        early_stopping_rounds=10,
        eval_metric='aucpr',
        eval_set=[(X_test, y_test)])

# Confusion matrix:
plot_confusion_matrix(clf,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels = ['Churned', 'Not churned'])

# Model predictions:
y_pred = clf.predict(X_test)

# Precision: 79,64%
precision_score(y_test, y_pred)

# Accuracy: 79.16%
accuracy_score(y_test, y_pred)

# f1 score: 79%
f1_score(y_test, y_pred)


'''
###########
Grid Search
###########
'''

# Grid dictionary:
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [1, 3, 5]
    }

optimal_params = GridSearchCV(
                estimator = xgb.XGBClassifier(objective='binary:classifier',
                                              subsample=0.9,
                                              colsample_bytree = 0.5),
                param_grid = param_grid,
                scoring = 'roc_auc',
                verbose = 0,
                n_jobs = 10,
                cv=3
                )

# Context Manager to avoid IPython parallelism error:
with parallel_backend('threading', n_jobs=2):   
    optimal_params.fit(X_test, y_test)

# Show optimal parameters:
optimal_params.best_params_

'''
###############
Optimized model
###############
'''
# Re-moddel with optimized parameters:
clf = xgb.XGBClassifier(objective='binary:logistic',
                        missing = 1,
                        gamma=0,
                        learning_rate=0.1, 
                        max_dept = 3,
                        reg_lambda = 0,
                        scale_pos_weight = 1)
# Fit the new model:
clf.fit(X_train,
        y_train,
        verbose = True,
        early_stopping_rounds=10,
        eval_metric='aucpr',
        eval_set=[(X_test, y_test)])

# New confusion matrix:
plot_confusion_matrix(clf,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels = ['Churned', 'Not churned'])

# New model predictions:
y_pred = clf.predict(X_test)

# Precision: 80.3% (v.s. old 79,64%)
precision_score(y_test, y_pred)

# Accuracy: 79.96% (v.s. old 79.16%)
accuracy_score(y_test, y_pred)

# f1 score: 80% (v.s. old 79%)
f1_score(y_test, y_pred)