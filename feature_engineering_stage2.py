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
    df_output['Fixture ID'] = df_sort['Target Fixture ID']
    df_output['Result Indicator'] = df_sort['Result Indicator']
    
    return df_output



def combining_fixture_id(df):
    #iterating over each opponent row to add to the previous
    odd_list = []
    for x in range(1, len(df)+1, 2):
        odd_list.append(x)
    even_list = []
    for x in range(0, len(df)-1, 2):
        even_list.append(x)
        
    team_df = df.drop(df.index[odd_list])
    team_df = team_df.reset_index(drop=True)
    opponent_df = df.drop(df.index[even_list])
    opponent_df = opponent_df.reset_index(drop=True)
    
    df_output = pd.DataFrame({})
    df_output['Team Av Shots Diff'] = team_df['Av Shots Diff']
    df_output['Team Av Shots Inside Box Diff'] = team_df['Av Shots Inside Box Diff']
    df_output['Team Av Fouls Diff'] = team_df['Av Fouls Diff']
    df_output['Team Av Corners Diff'] = team_df['Av Corners Diff']
    df_output['Team Av Possession Diff'] = team_df['Av Possession Diff']
    df_output['Team Av Pass Accuracy Diff'] = team_df['Av Pass Accuracy Diff']
    df_output['Team Av Goal Diff'] = team_df['Av Goal Difference']
    df_output['Opponent Av Shots Diff'] = opponent_df['Av Shots Diff']
    df_output['Opponent Av Shots Inside Box Diff'] = opponent_df['Av Shots Inside Box Diff']
    df_output['Opponent Av Fouls Diff'] = opponent_df['Av Fouls Diff']
    df_output['Opponent Av Corners Diff'] = opponent_df['Av Corners Diff']
    df_output['Opponent Av Possession Diff'] = opponent_df['Av Possession Diff']
    df_output['Opponent Av Pass Accuracy Diff'] = opponent_df['Av Pass Accuracy Diff']
    df_output['Opponent Av Goal Diff'] = opponent_df['Av Goal Difference']
    df_output['Fixture ID'] = team_df['Fixture ID']
    df_output['Team Result Indicator'] = team_df['Result Indicator']
    df_output['Opponent Result Indicator'] = opponent_df['Result Indicator']
    
    return df_output


def creating_ml_df(df):
    modified_df = mod_df(df)
    comb_df = combining_fixture_id(modified_df)
    return comb_df
        


#creating and saving the ml dataframe with a 5 game sliding average.
df_for_ml_5_v2 = creating_ml_df(df_ml_5)
with open('2019_prem_generated_clean/2019_prem_df_for_ml_5_v2.txt', 'wb') as myFile:
    pickle.dump(df_for_ml_5_v2, myFile)

#creating and saving the ml dataframe with a 10 game sliding average.
df_for_ml_10_v2 = creating_ml_df(df_ml_10)
with open('2019_prem_generated_clean/2019_prem_df_for_ml_10_v2.txt', 'wb') as myFile:
    pickle.dump(df_for_ml_10_v2, myFile)





# =============================================================================
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
# =============================================================================














# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')