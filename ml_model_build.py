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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold

plt.close('all')

#------------------------------- ML MODEL BUILD -------------------------------


with open('2019_prem_generated_clean/2019_prem_df_for_ml_5_v2.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('2019_prem_generated_clean/2019_prem_df_for_ml_10_v2.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)
    
    
#----------------------------- ML MODEL EVALUATION ----------------------------

#this section contains a series of functions which may be used for model evaluation

def pred_proba_plot(clf, x, y, cv=5, no_iter=5, no_bins=25, x_min=0.5, x_max=1, output_progress=True, classifier=''):
    '''
    This function outputs a histogram which informs the user about the number of correct and incorrect predictions given an outputed model probability on the test dataset. This give an indication as to whether the predicted proabilities may be trusted. 
    --Variables--
    clf: instantiated model with hyperparemeters
    x: feature array
    y: target array
    cv: no. cross-validation iterations
    no_iter: number of iterations of cross-validation
    no_bins: number of bins of histogram
    x_min: min x on histogram plot
    x_max: max x on histogram plot
    output_progress: print no. iterations complete to console
    '''
    y_dup = []
    correct_guess_pred = []
    incorrect_guess_pred = []
    for i in range(no_iter):
        if output_progress:
            if i % 2 == 0:
                print(f'completed {i} iterations')
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        y_pred_cv = cross_val_predict(clf, x, y, cv=skf)
        y_pred_proba_cv = cross_val_predict(clf, x, y, cv=skf, method='predict_proba')
        y_dup.append(list(y))
        for i in range(len(y_pred_cv)):
            if y_pred_cv[i] == list(y)[i]:
                correct_guess_pred.append(max(y_pred_proba_cv[i]))
            if y_pred_cv[i] != list(y)[i]:
                incorrect_guess_pred.append(max(y_pred_proba_cv[i]))         
    bins = np.linspace(x_min, x_max, no_bins)
    fig, ax = plt.subplots()
    ax.hist(incorrect_guess_pred, bins, alpha=0.5, edgecolor='#1E212A', color='red', label='Incorrect Prediction')
    ax.hist(correct_guess_pred, bins, alpha=0.5, edgecolor='#1E212A', color='green', label='Correct Prediction')
    ax.legend()
    fig.suptitle(f'{classifier} - Iterated {no_iter} Times', y=0.96, fontsize=16, fontweight='bold');
    ax.set(ylabel='Number of Occurences',
            xlabel='Prediction Probability')
    return fig



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
    
    return clf


ml_10_rand_forest = rand_forest_train(df_ml_10)
ml_5_rand_forest = rand_forest_train(df_ml_5)

with open('ml_models/random_forest_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_rand_forest, myFile)

with open('ml_models/random_forest_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_rand_forest, myFile)


#---------- MODEL EVALUATION ----------

#cross validation
cv_score_av = round(np.mean(cross_val_score(ml_10_rand_forest, x_10, y_10, cv=5))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_rand_forest, x_5, y_5, cv=5))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')

#prediction probability plots
#fig = pred_proba_plot(ml_10_rand_forest, x_10, y_10, no_iter=5, no_bins=36, x_min=0.3, classifier='Random Forest (ml_10)')
#fig.savefig('figures/random_forest_pred_proba_ml10_50iter.png')

#fig = pred_proba_plot(ml_5_rand_forest, x_5, y_5, no_iter=50, no_bins=35, x_min=0.3, classifier='Random Forest (ml_5)')
#fig.savefig('figures/random_forest_pred_proba_ml5_50iter.png')





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
fig = pred_proba_plot(ml_10_svm, x_10, y_10, no_iter=50, no_bins=35, x_min=0.3, classifier='Support Vector Machine (ml_10)')
fig.savefig('figures/svm_pred_proba_ml10_50iter.png')

fig = pred_proba_plot(ml_5_svm, x_5, y_5, no_iter=50, no_bins=35, x_min=0.3, classifier='Support Vector Machine (ml_5)')
fig.savefig('figures/svm_pred_proba_ml5_50iter.png')





print('\nK NEAREST NEIGHBORS\n')
#----------------------------- K NEAREST NEIGHBORS ----------------------------


def k_nearest_neighbor_train(df):
    
    #create features matrix
    x = df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
    y = df['Team Result Indicator']
    
    #split into training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
      
    #instantiate the K Nearest Neighbor class
    clf = KNeighborsClassifier(n_neighbors=28)
    
    #train the model
    clf.fit(x_train, y_train)
    
    #training data
    train_data_score = round(clf.score(x_train, y_train) * 100, 1)
    print(f'Training data score = {train_data_score}%')
    
    #test data
    test_data_score = round(clf.score(x_test, y_test) * 100, 1)
    print(f'Test data score = {test_data_score}% \n')
    
    return clf


ml_10_knn = k_nearest_neighbor_train(df_ml_10)
ml_5_knn = k_nearest_neighbor_train(df_ml_5)

with open('ml_models/knn_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_knn, myFile)

with open('ml_models/knn_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_knn, myFile)


# --------------- TESTING N_NEIGHBORS PARAM ---------------

df = df_ml_10

#create features matrix
x = df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y = df['Team Result Indicator']

#split into training data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    

test_accuracy = []
for n in range(1, 50, 1):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(x_train, y_train)
    test_accuracy.append(round(clf.score(x_test, y_test) * 100, 1))

fig, ax = plt.subplots()
ax.plot(range(1,50, 1), test_accuracy)


#---------- MODEL EVALUATION ----------

#cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True)

cv_score_av = round(np.mean(cross_val_score(ml_10_knn, x_10, y_10, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_knn, x_5, y_5, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')


#prediction probability plots
fig = pred_proba_plot(ml_10_knn, x_10, y_10, no_iter=50, no_bins=18, x_min=0.3, classifier='Nearest Neighbor (ml_10)')
fig.savefig('figures/knn_pred_proba_ml10_50iter.png')

fig = pred_proba_plot(ml_5_knn, x_5, y_5, no_iter=50, no_bins=18, x_min=0.3, classifier='Nearest Neighbor (ml_5)')
fig.savefig('figures/knn_pred_proba_ml5_50iter.png')


    
# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')