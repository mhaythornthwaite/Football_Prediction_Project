# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:59:55 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

#!/usr/bin/python
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep)


from ml_functions.ml_model_eval import pred_proba_plot, plot_cross_val_confusion_matrix
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict


plt.close('all')

#------------------------------- ML MODEL BUILD -------------------------------

#importing the data and creating the feature dataframe and target series

with open('../2019_prem_generated_clean/2019_prem_df_for_ml_5_v2.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('../2019_prem_generated_clean/2019_prem_df_for_ml_10_v2.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)


x_10 = df_ml_10.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_10 = df_ml_10['Team Result Indicator']

x_5 = df_ml_5.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_5 = df_ml_5['Team Result Indicator']


print('\nRANDOM FOREST\n')
#------------------------------- RANDOM FOREST --------------------------------

def rand_forest_train(df):

    #create features matrix
    x = df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
    y = df['Team Result Indicator']
    
    #instantiate the random forest class
    clf = RandomForestClassifier()
    
    #split into training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    #train the model
    clf.fit(x_train, y_train)
    
    #training data
    train_data_score = round(clf.score(x_train, y_train) * 100, 1)
    print(f'Training data score = {train_data_score}%')
    
    #test data
    test_data_score = round(clf.score(x_test, y_test) * 100, 1)
    print(f'Test data score = {test_data_score}% \n')
    
    return clf, x_train, x_test, y_train, y_test


ml_10_rand_forest, x10_train, x10_test, y10_train, y10_test = rand_forest_train(df_ml_10)
ml_5_rand_forest, x5_train, x5_test, y5_train, y5_test = rand_forest_train(df_ml_5)

with open('ml_models/random_forest_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_rand_forest, myFile)

with open('ml_models/random_forest_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_rand_forest, myFile)


#---------- MODEL EVALUATION ----------

#cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True)

cv_score_av = round(np.mean(cross_val_score(ml_10_rand_forest, x_10, y_10, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_rand_forest, x_5, y_5, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')


#prediction probability plots
#fig = pred_proba_plot(ml_10_rand_forest, x_10, y_10, no_iter=5, no_bins=36, x_min=0.3, classifier='Random Forest (ml_10)')
#fig.savefig('figures/random_forest_pred_proba_ml10_50iter.png')

#fig = pred_proba_plot(ml_5_rand_forest, x_5, y_5, no_iter=50, no_bins=35, x_min=0.3, classifier='Random Forest (ml_5)')
#fig.savefig('figures/random_forest_pred_proba_ml5_50iter.png')


#plot confusion matrix - modified to take cross-val results.
plot_cross_val_confusion_matrix(ml_10_rand_forest, x_10, y_10, display_labels=('team loses', 'draw', 'team wins'), title='Random Forest Confusion Matrix ML10', cv=skf)
plt.savefig('figures\ml_10_confusion_matrix_cross_val_random_forest.png')

plot_cross_val_confusion_matrix(ml_5_rand_forest, x_5, y_5, display_labels=('team loses', 'draw', 'team wins'), title='Random Forest Confusion Matrix ML5', cv=skf)
plt.savefig('figures\ml_5_confusion_matrix_cross_val_random_forest.png')


# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')