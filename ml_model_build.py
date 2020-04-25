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


#------------------------------- ML MODEL BUILD -------------------------------


with open('2019_prem_generated_clean/2019_prem_df_for_ml_5.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('2019_prem_generated_clean/2019_prem_df_for_ml_10.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)


#test change






# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')