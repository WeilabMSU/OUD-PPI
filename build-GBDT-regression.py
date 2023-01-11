# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:43:10 2019

@author: fenghon1
"""

import numpy as np
import argparse
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
import scipy.stats as stats
import os
from statistics import mean
import pickle
import sys
import argparse

parser = argparse.ArgumentParser(description='building models for OUD')
parser.add_argument('--save_model_name', type=str, help='the name for the built model')
parser.add_argument('--feature_path', type=str,help='feature_path')
parser.add_argument('--label_path', type=str, help='label_path')
args = parser.parse_args()

save_model_name = args.save_model_name
feature_path = args.feature_path
label_path = args.label_path


def read_dataset(feature_file, label_file):
    df_X = np.load(feature_file)
    df_y = pd.read_csv(label_file,header=None, index_col=False)
    X = df_X
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y

ml_method="GradientBoostingRegressor"
model_path = 'path-models'
if not os.path.exists(model_path):
    os.mkdir(model_path)

X_train, y_train = read_dataset(feature_path, label_path)
y_train=np.ravel(y_train)

i=20000; j=7 ;k=5; m=8;lr=0.002
#i=100; j=7 ;k=5; m=8;lr=0.002
clf=globals()["%s"%ml_method](n_estimators=i,max_depth=j,min_samples_split=k,learning_rate=lr,subsample=0.1*m,max_features='sqrt')

clf.fit(X_train, y_train)

filename = '%s/reg-model-%s.sav'%(model_path,save_model_name)
pickle.dump(clf, open(filename, 'wb'))

