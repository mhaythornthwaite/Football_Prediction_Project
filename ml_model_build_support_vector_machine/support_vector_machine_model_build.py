# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:28:28 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

#!/usr/bin/python
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep)


from ml_functions.ml_model_eval import pred_proba_plot
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
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



print('\nSUPPORT VECTOR MACHINES\n')
#--------------------------- SUPPORT VECTOR MACHINE ---------------------------


def svm_train(df):
    
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
    
    #training data
    train_data_score = round(clf.score(x_train, y_train) * 100, 1)
    print(f'Training data score = {train_data_score}%')
    
    #test data
    test_data_score = round(clf.score(x_test, y_test) * 100, 1)
    print(f'Test data score = {test_data_score}% \n')
    
    return clf


ml_10_svm = svm_train(df_ml_10)
ml_5_svm = svm_train(df_ml_5)


with open('ml_models/svm_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_svm, myFile)

with open('ml_models/svm_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_svm, myFile)


# --------------- TESTING C PARAM ---------------

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
    

#---------- MODEL EVALUATION ----------

#cross validation
cv_score_av = round(np.mean(cross_val_score(ml_10_svm, x_10, y_10, cv=5))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_svm, x_5, y_5, cv=5))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')

#prediction probability plots
#fig = pred_proba_plot(ml_10_svm, x_10, y_10, no_iter=50, no_bins=35, x_min=0.3, classifier='Support Vector Machine (ml_10)')
#fig.savefig('figures/svm_pred_proba_ml10_50iter.png')

#fig = pred_proba_plot(ml_5_svm, x_5, y_5, no_iter=50, no_bins=35, x_min=0.3, classifier='Support Vector Machine (ml_5)')
#fig.savefig('figures/svm_pred_proba_ml5_50iter.png')



    
# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')

