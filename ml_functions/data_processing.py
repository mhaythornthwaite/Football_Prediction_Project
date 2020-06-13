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
  

#------------------------------- DATA PROCESSING ------------------------------


def scale_df(df):
    '''
    This function will use the preprocessing function in sklearn to rescale each feature to haze zero mean and unit vector (mean=0, variance=1). Output is df instead of np array

    Parameters
    ----------
    df : pandas DataFrame

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
    
    print(scaled_df.head)
    
    return scaled_df
        
        
        
        
        