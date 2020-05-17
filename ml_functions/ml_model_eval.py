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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  

#----------------------------- ML MODEL EVALUATION ----------------------------

#this section contains a series of functions which may be used for model evaluation

def pred_proba_plot(clf, x, y, cv=5, no_iter=5, no_bins=25, x_min=0.5, x_max=1, output_progress=True, classifier=''):
    '''
    Parameters
    ----------
    clf : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data..
    x : array-like
        The data to fit. Can be, for example a list, or an array at least 2d..
    y : array-like
        The target variable to try to predict in the case of
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.. The default is 5.
    no_iter : int, optional
        number of iterations of cross-validation. The default is 5.
    no_bins : int, optional
        number of bins of histogram. The default is 25.
    x_min : int, optional
        min x on histogram plot. The default is 0.5.
    x_max : in, optional
        max x on histogram plot. The default is 1.
    output_progress : display, optional
        print no. iterations complete to console. The default is True.
    classifier : string, optional
        classifier used, will be input to title. The default is ''.

    Returns
    -------
    fig : histogram display of correcly predicted results against incorrectly given results given the outputed probability of the classifier.
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




def plot_cross_val_confusion_matrix(clf, x, y, display_labels='', title='', cv=5):
    '''
    Parameters
    ----------
    clf : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    x : array-like
        The data to fit. Can be, for example a list, or an array at least 2d..
    y : array-like
        The target variable to try to predict in the case of
    supervised learning.
    display_labels : ndarray of shape (n_classes,), optional
        display labels for plot. The default is ''.
    title : string, optional
        Title to be displayed at top of plot. The default is ''.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.. The default is 5.

    Returns
    -------
    display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
    '''
    
    y_pred = cross_val_predict(clf, x, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    fig = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig.plot(cmap=plt.cm.Blues)
    fig.ax_.set_title(title)
    return fig


   
# ----------------------------------- END -------------------------------------
