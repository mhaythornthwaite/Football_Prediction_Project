# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:08:41 2020

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

import pickle
import numpy as np
import pandas as pd
from ml_functions.ml_model_eval import pred_proba_plot, plot_cross_val_confusion_matrix, plot_learning_curve
from sklearn.ensemble import RandomForestClassifier
from ml_functions.data_processing import scale_df
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict


#------------------------------- INPUT VARIABLES ------------------------------

fixtures_saved_name = '2019_2020_premier_league_fixtures_df.csv'


#---------------------- ALTERNATIVE FEATURE ENGINEERING -----------------------

fixtures_clean = pd.read_csv(f'../prem_clean_fixtures_and_dataframes/{fixtures_saved_name}')

with open('../prem_clean_fixtures_and_dataframes/2019_2020_prem_df_for_ml_10_v2.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

alt_df = pd.DataFrame({})  

alt_df['Shots Diff'] = df_ml_10['Team Av Shots Diff'] - df_ml_10['Opponent Av Shots Diff'] 
alt_df['Shots Box Diff'] = df_ml_10['Team Av Shots Inside Box Diff'] - df_ml_10['Opponent Av Shots Inside Box Diff'] 
alt_df['Corners Diff'] = df_ml_10['Team Av Corners Diff'] - df_ml_10['Opponent Av Corners Diff'] 
alt_df['Possession Diff'] = df_ml_10['Team Av Possession Diff'] - df_ml_10['Opponent Av Possession Diff']
alt_df['Pass Diff'] = df_ml_10['Team Av Pass Accuracy Diff'] - df_ml_10['Opponent Av Pass Accuracy Diff']
alt_df['Goal Diff'] = df_ml_10['Team Av Goal Diff'] - df_ml_10['Opponent Av Goal Diff']
alt_df['Fixture ID'] = df_ml_10['Fixture ID']
alt_df['Team Result Indicator'] = df_ml_10['Team Result Indicator']
alt_df['Opponent Result Indicator'] = df_ml_10['Opponent Result Indicator']

x_10 = alt_df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_10 = alt_df['Team Result Indicator']


#------------------------------- ML MODEL BUILD -------------------------------

x = alt_df
y = alt_df['Team Result Indicator']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_fixture_id = x_test['Fixture ID']

x_train = x_train.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
x_test = x_test.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)


ml_alt_df = RandomForestClassifier(max_depth=4, max_features=4, n_estimators=120)
ml_alt_df.fit(x_train, y_train)

ml_df_10 = RandomForestClassifier(max_depth=4, max_features=4, n_estimators=120)
ml_df_10.fit(x_train, y_train)


# ---------- CROSS VALIDATION ----------

skf = StratifiedKFold(n_splits=5, shuffle=True)

cv_score_av = round(np.mean(cross_val_score(ml_alt_df, x_10, y_10, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')


# ---------- FEATURE IMPORTANCE ----------

fi_ml_10 = pd.DataFrame({'feature': list(x_train.columns),
                         'importance': ml_alt_df.feature_importances_}).sort_values('importance', ascending = False)


# ---------- CONFUSION MATRIX PLOTS ----------

#modified to take cross-val results.

plot_cross_val_confusion_matrix(ml_alt_df, 
                                x_10, 
                                y_10, 
                                display_labels=('team loses', 'draw', 'team wins'), 
                                title='Random Forest Confusion Matrix ML10', 
                                cv=skf)


predictions_alt_df = ml_alt_df.predict_proba(x_test)
predictions_df_ml_10 = ml_df_10.predict_proba(x_test)


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')