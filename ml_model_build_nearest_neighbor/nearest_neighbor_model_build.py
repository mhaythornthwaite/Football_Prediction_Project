# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:28:00 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

from ml_functions.ml_model_eval import pred_proba_plot
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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
#fig.savefig('figures/knn_pred_proba_ml10_50iter.png')

#fig = pred_proba_plot(ml_5_knn, x_5, y_5, no_iter=50, no_bins=18, x_min=0.3, classifier='Nearest Neighbor (ml_5)')
#fig.savefig('figures/knn_pred_proba_ml5_50iter.png')


    
# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')