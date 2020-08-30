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

from ml_functions.data_processing import scale_df
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score

plt.close('all')


#------------------------------- INPUT VARIABLES ------------------------------

df_5_saved_name = '2019_prem_df_for_ml_5_v2.txt'
df_10_saved_name = '2019_prem_df_for_ml_10_v2.txt'

grid_search = False
save_grid_search_fig = False

create_final_model = True

#------------------------------- ML MODEL BUILD -------------------------------

#importing the data and creating the feature dataframe and target series

with open(f'../prem_clean_fixtures_and_dataframes/{df_5_saved_name}', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open(f'../prem_clean_fixtures_and_dataframes/{df_10_saved_name}', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

#scaling dataframe to make all features to have zero mean and unit vector.
df_ml_10 = scale_df(df_ml_10, list(range(14)), [14,15,16])
df_ml_5 = scale_df(df_ml_5, list(range(14)), [14,15,16])

#creating feautes and labels df for df_10
x_10 = df_ml_10.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_10 = df_ml_10['Team Result Indicator']

#creating feautes and labels df for df_5
x_5 = df_ml_5.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_5 = df_ml_5['Team Result Indicator']


#---------------------------- MULTILAYER PERCEPTRON ---------------------------

#split into training data and test data
x_train, x_test, y_train, y_test = train_test_split(x_10, y_10, test_size=0.2)

#instantiate the MLP classifier, and fit to the data
clf = MLPClassifier(hidden_layer_sizes=(18, 12), activation='logistic', random_state=0, max_iter=5000)
clf.fit(x_train, y_train)

#printing the cross-validation accuracy score
cv_score_av = round(np.mean(cross_val_score(clf, x_10, y_10))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')


#--------------------------------- GRID SEARCH --------------------------------

if grid_search:
    #creating the list of tuples to test the length of hidden layers
    hidden_layer_test = []
    for i in range(6,20,2):
        a = list(range(6,30,4))
        b = [i] * len(a)    
        c = list(zip(a, b))
        hidden_layer_test.extend(c)    
        
    param_grid_grad = [{'hidden_layer_sizes':hidden_layer_test}]
    
    #mlp gridsearch 
    grid_search_grad = GridSearchCV(clf, param_grid_grad, cv=5, scoring = 'accuracy', return_train_score = True)
    grid_search_grad.fit(x_10, y_10)

    #Output best Cross Validation score and parameters from grid search
    print('\n', 'Gradient Best Params: ' , grid_search_grad.best_params_)
    print('Gradient Best Score: ' , grid_search_grad.best_score_ , '\n')

    print(grid_search_grad.cv_results_['mean_test_score'])


# ---------- PLOTTING THE DATA ----------

if grid_search:
    #populating df with x, y, z data
    matrix_plot_data = pd.DataFrame({})
    matrix_plot_data['x'] = list(zip(*hidden_layer_test))[0]
    matrix_plot_data['y'] = list(zip(*hidden_layer_test))[1]
    matrix_plot_data['z'] = grid_search_grad.cv_results_['mean_test_score']
    
    #transforming z list into matrix format
    Z = matrix_plot_data.pivot_table(index='x', columns='y', values='z').T.values
    
    #getting x and y axis
    X_unique = np.sort(matrix_plot_data.x.unique())
    Y_unique = np.sort(matrix_plot_data.y.unique())
    
    #instantiating figure and plotting heatmap with seaborn
    fig, ax = plt.subplots()
    im = sns.heatmap(Z, annot=True,  linewidths=.5)
    
    #labelling x and y and title
    ax.set_xticklabels(X_unique)
    ax.set_yticklabels(Y_unique)
    ax.set(xlabel='Hidden Layer 1 Length',
            ylabel='Hidden Layer 2 Length');
    fig.suptitle('Cross Val Accuracy', y=0.95, fontsize=16, fontweight='bold');
    
    if save_grid_search_fig:
        fig.savefig('figures/testing_hidden_layer_lengths.png')    


#--------------------------------- FINAL MODEL --------------------------------

#in this section we will take the learnings from the hyperparameter testing above and train a final model using 100% of the data. This model may then be used for predictions going forward.

if create_final_model:
    
    #intantiating and training the df_5 network
    ml_5_mlp = MLPClassifier(hidden_layer_sizes=(18, 12), activation='relu', random_state=0, max_iter=5000)
    ml_5_mlp.fit(x_5, y_5)
    
    #intantiating and training the df_10 network
    ml_10_mlp = MLPClassifier(hidden_layer_sizes=(18, 12), activation='relu', random_state=0, max_iter=5000)
    ml_10_mlp.fit(x_10, y_10)
    
    with open('ml_models/mlp_model_5.pk1', 'wb') as myFile:
        pickle.dump(ml_5_mlp, myFile)

    with open('ml_models/mlp_model_10.pk1', 'wb') as myFile:
        pickle.dump(ml_10_mlp, myFile)


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
