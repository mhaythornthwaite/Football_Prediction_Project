# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:04:11 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pandas as pd
import pickle
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm


#------------------------------- ML MODEL BUILD -------------------------------


with open('2019_prem_generated_clean/2019_prem_df_for_ml_5_v2.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('2019_prem_generated_clean/2019_prem_df_for_ml_10_v2.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)


print('\nRANDOM FOREST\n')
#------------------------------- RANDOM FOREST --------------------------------

def rand_forest_train(df):
    '''
    Parameters
    ----------
    df : Pandas df
        df in the format of df_ml_10 or df_ml_5.

    Returns
    -------
    Trained model as well as printing the training and test data set accuracy.

    '''
    #create features matrix
    x = df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
    y = df['Team Result Indicator']
    
    #instantiate the random forest class
    clf = RandomForestClassifier()
    
    #split into training data and test data
    np.random.seed(0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1)
    
    #train the model
    clf.fit(x_train, y_train)
    
    #making predictions on the test dataset
    y_pred = clf.predict(x_test)
    
    #training data
    train_data_score = round(clf.score(x_train, y_train) * 100, 1)
    print(f'Training data score = {train_data_score}%')
    
    #test data
    test_data_score = round(clf.score(x_test, y_test) * 100, 1)
    print(f'Test data score = {test_data_score}% \n')
    
    return clf

ml_10_rand_forest = rand_forest_train(df_ml_10)
ml_5_rand_forest = rand_forest_train(df_ml_5)


with open('ml_models/random_forest_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_rand_forest, myFile)

with open('ml_models/random_forest_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_rand_forest, myFile)


print('\nSUPPORT VECTOR MACHINES\n')
#--------------------------- SUPPORT VECTOR MACHINE ---------------------------


def svm_train(df):
    
    #create features matrix
    x = df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
    y = df['Team Result Indicator']
    
    #instantiate the SVM class
    clf = svm.SVC()
    
    #split into training data and test data
    np.random.seed(0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1)
    
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




# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')