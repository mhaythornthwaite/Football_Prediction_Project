# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:15:48 2020

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

#!/usr/bin/python
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep)

import time
start=time.time()

import pandas as pd
import pickle
import numpy as np
import math
from ml_functions.data_processing import scale_df
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict


with open('../prem_clean_fixtures_and_dataframes/2019_2020_prem_df_for_ml_10_v2.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)
    
    
#scaling dataframe to make all features to have zero mean and unit vector.
df_ml_10 = scale_df(df_ml_10, list(range(14)), [14,15,16])

x_10 = df_ml_10.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_10 = df_ml_10['Team Result Indicator']

X_train, X_test, y_train, y_test = train_test_split(x_10, y_10, test_size=0.2)


clf_rf = pickle.load(open('../ml_model_build_random_forest/ml_models/random_forest_model_10.pk1', 'rb'))

clf_svm = pickle.load(open('../ml_model_build_support_vector_machine/ml_models/svm_model_10.pk1', 'rb'))

clf_knn = pickle.load(open('../ml_model_build_nearest_neighbor/ml_models/knn_model_10.pk1', 'rb'))

clf_mlp = pickle.load(open('../ml_model_build_deep_learning/ml_models/mlp_model_10.pk1', 'rb'))


#create a dictionary of our models
estimators=[('rf', clf_rf), ('svm', clf_svm), ('knn', clf_knn), ('mlp', clf_mlp)]
#estimators=[('rf', clf_rf), ('knn', clf_knn), ('mlp', clf_mlp)]
#estimators=[('rf', clf_rf), ('knn', clf_knn)]

#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')

#fit model to training data
ensemble.fit(X_train, y_train)


#cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True)

cv_score_av = round(np.mean(cross_val_score(ensemble, x_10, y_10, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')



# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
