# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:04:11 2020

@author: mhayt
"""

#-------------------------------- API-FOOTBALL --------------------------------


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
  

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




   
# ----------------------------------- END -------------------------------------
