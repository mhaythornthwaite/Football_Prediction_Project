# -*- coding: utf-8 -*-
"""
Created on Sat May  2 12:09:22 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pandas as pd
import pickle
import numpy as np


#----------------------------- FEATURE ENGINEERING ----------------------------

with open('2019_prem_generated_clean/2019_prem_df_for_ml_5.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('2019_prem_generated_clean/2019_prem_df_for_ml_10.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

def sort_df(df):
    df = df.sort_values('Target Fixture ID')
    df = df.reset_index(drop=True)
    return df

df_ml_10_sort = sort_df(df_ml_10)
df_ml_5_sort = sort_df(df_ml_5)



def mod_df(df):
    df_sort = df.sort_values('Target Fixture ID')
    df_sort = df_sort.reset_index(drop=True)
    
    df_output = pd.DataFrame({})
    #creating our desired features
    df_output['Av Shots Diff'] = df_sort['Team Av Shots'] - df_sort['Opponent Av Shots']
    df_output['Av Shots Inside Box Diff'] = df_sort['Team Av Shots Inside Box'] - df_sort['Opponent Av Shots Inside Box']
    df_output['Av Fouls Diff'] = df_sort['Team Av Fouls'] - df_sort['Opponent Av Fouls']
    df_output['Av Corners Diff'] = df_sort['Team Av Corners'] -df_sort['Opponent Av Corners']
    df_output['Av Possession Diff'] = df_sort['Team Av Possession'] - df_sort['Opponent Av Possession']
    df_output['Av Pass Accuracy Diff'] = df_sort['Team Av Pass Accuracy'] - df_sort['Opponent Av Pass Accuracy']
    df_output['Av Goal Difference'] = df_sort['Team Av Goals'] - df_sort['Opponent Av Goals']
    
    return df_output

df_ml_10_mod = mod_df(df_ml_10)


# =============================================================================
# 
# 
#     df_ready_for_ml = pd.DataFrame({})  
#     df_ready_for_ml['Team Av Shots'] = t_total_shots
#     df_ready_for_ml['Team Av Shots Inside Box'] = t_shots_inside_box
#     df_ready_for_ml['Team Av Fouls'] = t_fouls
#     df_ready_for_ml['Team Av Corners'] = t_corners
#     df_ready_for_ml['Team Av Possession'] = t_posession
#     df_ready_for_ml['Team Av Pass Accuracy'] = t_pass_accuracy
#     df_ready_for_ml['Team Av Goals'] = t_goals
#     df_ready_for_ml['Opponent Av Shots'] = o_total_shots
#     df_ready_for_ml['Opponent Av Shots Inside Box'] = o_shots_inside_box
#     df_ready_for_ml['Opponent Av Fouls'] = o_fouls
#     df_ready_for_ml['Opponent Av Corners'] = o_corners
#     df_ready_for_ml['Opponent Av Possession'] = o_posession
#     df_ready_for_ml['Opponent Av Goals'] = o_goals
#     df_ready_for_ml['Opponent Av Pass Accuracy'] = o_pass_accuracy
#     df_ready_for_ml['Team Goal Target'] = t_goals_target
#     df_ready_for_ml['Opponent Goal Target'] = o_goals_target
#     df_ready_for_ml['Target Fixture ID'] = fix_id
#     
# 
# =============================================================================













# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')