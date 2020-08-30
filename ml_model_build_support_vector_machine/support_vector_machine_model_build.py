# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:28:28 2020

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

from ml_functions.ml_model_eval import pred_proba_plot, plot_cross_val_confusion_matrix, plot_learning_curve
from ml_functions.data_processing import scale_df
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

plt.close('all')


#------------------------------- ML MODEL BUILD -------------------------------

#importing the data and creating the feature dataframe and target series

with open('../prem_clean_fixtures_and_dataframes/2019_prem_df_for_ml_5_v2.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('../prem_clean_fixtures_and_dataframes/2019_prem_df_for_ml_10_v2.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

#scaling dataframe to make all features to have zero mean and unit vector.
df_ml_10 = scale_df(df_ml_10, list(range(14)), [14,15,16])
df_ml_5 = scale_df(df_ml_5, list(range(14)), [14,15,16])

x_10 = df_ml_10.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_10 = df_ml_10['Team Result Indicator']

x_5 = df_ml_5.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_5 = df_ml_5['Team Result Indicator']


#--------------------------- SUPPORT VECTOR MACHINE ---------------------------


def svm_train(df, print_result=True, print_result_label=''):
    
    #create features matrix
    x = df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
    y = df['Team Result Indicator']
    
    #split into training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    #default gamma value
    gamma = 1 / (14 * sum(x_train.var()))
    C = 1 / gamma
    
    #instantiate the SVM class
    clf = svm.SVC(kernel='rbf', C=3, probability=True)
    
    #train the model
    clf.fit(x_train, y_train)
    
    if print_result:
        print(print_result_label)
        #training data
        train_data_score = round(clf.score(x_train, y_train) * 100, 1)
        print(f'Training data score = {train_data_score}%')
        
        #test data
        test_data_score = round(clf.score(x_test, y_test) * 100, 1)
        print(f'Test data score = {test_data_score}% \n')
    
    return clf, x_train, x_test, y_train, y_test


ml_10_svm, x10_train, x10_test, y10_train, y10_test = svm_train(df_ml_10)
ml_5_svm, x5_train, x5_test, y5_train, y5_test = svm_train(df_ml_5)

#with open('ml_models/svm_model_5.pk1', 'wb') as myFile:
#    pickle.dump(ml_5_svm, myFile)

#with open('ml_models/svm_model_10.pk1', 'wb') as myFile:
#    pickle.dump(ml_10_svm, myFile)


# ---------- TESTING C PARAM ----------

expo_iter = np.square(np.arange(0.1, 10, 0.1))

def testing_c_parms(df, iterable):
    training_score_li = []
    test_score_li = []
    for c in iterable:
        x = df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
        y = df['Team Result Indicator']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1)
        clf = svm.SVC(kernel='rbf', C=c)
        clf.fit(x_train, y_train)
        train_data_score = round(clf.score(x_train, y_train) * 100, 1)
        test_data_score = round(clf.score(x_test, y_test) * 100, 1)
        training_score_li.append(train_data_score)
        test_score_li.append(test_data_score)
    return training_score_li, test_score_li
    
training_score_li, test_score_li = testing_c_parms(df_ml_10, expo_iter)

#from the plot below we can see that a c of around 3 is likely to be more optimal than 1
fig, ax = plt.subplots()
ax.plot(expo_iter, test_score_li)
    

# ---------- ENSEMBLE MODELLING ----------

#In this section we will combine the results of using the same algorithm but with different input data used to train the model. The features are still broadly the same but have been averaged over a different number of games df_ml_10 is 10 games, df_ml_5 is 5 games. 

#reducing fixtures in df_ml_5 to contain only the fixtures within df_ml_10 and training that new dataset
df_ml_5_dropto10 = df_ml_5.drop(list(range(0,50)))
ml_5_to10_svm, x5_to10_train, x5_to10_test, y5_to10_train, y5_to10_test = svm_train(df_ml_5_dropto10, print_result=False)

#making predictions using the two df inputs independantly
y_pred_ml10 = ml_10_svm.predict(x10_test)
y_pred_ml5to10 = ml_5_to10_svm.predict(x10_test)

#making probability predictions on each of the datasets independantly
pred_proba_ml10 = ml_10_svm.predict_proba(x10_test)
pred_proba_ml5_10 = ml_5_to10_svm.predict_proba(x10_test)

#combining independant probabilities and creating combined class prediction
pred_proba_ml5and10 = (np.array(pred_proba_ml10) + np.array(pred_proba_ml5_10)) / 2.0
y_pred_ml5and10 = np.argmax(pred_proba_ml5and10, axis=1)

#accuracy score variables
y_pred_ml10_accuracy = round(accuracy_score(y10_test, y_pred_ml10), 3) * 100
y_pred_ml5to10_accuracy = round(accuracy_score(y10_test, y_pred_ml5to10), 3) * 100
y_pred_ml5and10_accuracy = round(accuracy_score(y10_test, y_pred_ml5and10), 3) * 100

print('ENSEMBLE MODEL TESTING')
print(f'Accuracy of df_10 alone = {y_pred_ml10_accuracy}%')
print(confusion_matrix(y10_test, y_pred_ml10), '\n')
print(f'Accuracy of df_5 alone = {y_pred_ml5to10_accuracy}%')
print(confusion_matrix(y10_test, y_pred_ml5to10), '\n')
print(f'Accuracy of df_5 and df_10 combined = {y_pred_ml5and10_accuracy}%')
print(confusion_matrix(y10_test, y_pred_ml5and10), '\n\n')


# ---------- MODEL EVALUATION ----------

#cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True)

cv_score_av = round(np.mean(cross_val_score(ml_10_svm, x_10, y_10, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_svm, x_5, y_5, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')


#prediction probability plots
fig = pred_proba_plot(ml_10_svm, x_10, y_10, no_iter=50, no_bins=35, x_min=0.3, classifier='Support Vector Machine (ml_10)')
#fig.savefig('figures/ml_10_svm_pred_proba.png')

fig = pred_proba_plot(ml_5_svm, x_5, y_5, no_iter=50, no_bins=35, x_min=0.3, classifier='Support Vector Machine (ml_5)')
#fig.savefig('figures/ml_5_svm_pred_proba.png')


#plot confusion matrix - modified to take cross-val results.
plot_cross_val_confusion_matrix(ml_10_svm, x_10, y_10, display_labels=('team loses', 'draw', 'team wins'), title='Support Vector Machine Confusion Matrix ML10', cv=skf)
#plt.savefig('figures\ml_10_confusion_matrix_cross_val_svm.png')

plot_cross_val_confusion_matrix(ml_5_svm, x_5, y_5, display_labels=('team loses', 'draw', 'team wins'), title='Support Vector Machine Confusion Matrix ML5', cv=skf)
#plt.savefig('figures\ml_5_confusion_matrix_cross_val_svm.png')


#plotting learning curves
plot_learning_curve(ml_10_svm, x_10, y_10, training_set_size=10, x_max=160, title='Learning Curve - Support Vector Machine DF_10', leg_loc=1)
#plt.savefig('figures\ml_10_svm_learning_curve.png')

plot_learning_curve(ml_5_svm, x_5, y_5, training_set_size=10, x_max=230, title='Learning Curve - Support Vector Machine DF_5', leg_loc=1)
#plt.savefig('figures\ml_5_svm_learning_curve.png')

    
# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
