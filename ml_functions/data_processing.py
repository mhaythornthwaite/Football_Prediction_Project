# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:15:43 2020

@author: mhayt
"""

#-------------------------------- API-FOOTBALL --------------------------------


import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
  

#------------------------------- DATA PROCESSING ------------------------------


def scale_df(df, scale, unscale):
    '''
    This function will use the preprocessing function in sklearn to rescale each feature to haze zero mean and unit vector (mean=0, variance=1). Output is df instead of np array

    Parameters
    ----------
    df : pandas DataFrame
        df to manipulate/scale
    scale : list
        list of column indices to process/scale
    unscale : list
        list of column indices to remain the same

    Returns
    -------
    Scaled df.

    '''
    
    scaled_np = preprocessing.scale(df)
    
    col_list = []
    for col in df.columns:
        col_list.append(col)
    
    scaled_df = pd.DataFrame(scaled_np)
    scaled_df.columns = col_list
    
    df1 = scaled_df.iloc[:, scale]
    df2 = df.iloc[:, unscale]
    
    final_df = pd.concat([df1, df2], axis=1, sort=False)
    
    return final_df

        

def scree_plot(pca_percentages, y_max=40):
    '''
    Input principle component percentages list and returns scree plot

    Parameters
    ----------
    pca_percentages : list
        principle component percentage variation.

    Returns
    -------
    fig : fig
        bar plot.

    '''
    
    #setting up variables
    n_components = len(pca_percentages)
    
    #instantiating figure
    fig, ax = plt.subplots()
    
    #plot bar component
    ax.bar(list(range(1, n_components+1, 1)), pca_percentages, color='paleturquoise', edgecolor='darkturquoise', zorder=0)
    
    #annotating with percentages
    for p in ax.patches:
        ax.annotate(f'{round(p.get_height(), 1)}%', (p.get_x() + 0.5, p.get_height() + 0.5))
    
    #plot line and points of each principle component
    ax.plot(list(range(1, n_components+1, 1)), pca_percentages, c='firebrick', zorder=1)
    ax.scatter(list(range(1, n_components+1, 1)), pca_percentages, c='firebrick', zorder=2)
    
    #Plotting details
    fig.suptitle('PCA Scree Plot', y=0.96, fontsize=16, fontweight='bold');
    ax.set(xlabel='Principle Components',
           ylabel='Percentage Variation');
    ax.set_ylim([0,y_max])
    
    return fig


