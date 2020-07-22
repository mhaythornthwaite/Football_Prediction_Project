# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:23:13 2020

@author: mhayt
"""


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pickle
from ml_functions.feature_engineering_functions import average_stats_df
from ml_functions.feature_engineering_functions import creating_ml_df

#----------------------------- FEATURE ENGINEERING ----------------------------


with open('2019_prem_generated_clean/2019_prem_all_stats_dict.txt', 'rb') as myFile:
    game_stats = pickle.load(myFile)


#creating a list with the team id in
team_list = []
for key in game_stats.keys():
    team_list.append(key)
team_list.sort()


#creating a dictionary with the team id as key and fixture id's as values
team_fixture_id_dict = {}
for team in team_list:
    fix_id_list = []
    for key in game_stats[team].keys():
        fix_id_list.append(key)
    fix_id_list.sort()
    sub_dict = {team:fix_id_list}
    team_fixture_id_dict.update(sub_dict)
        

#the list of fixtures was first home then away, we want them in chronological order so we need to sort them.
for team in team_fixture_id_dict:
    team_fixture_id_dict[team].sort()


#we can now iterate over the fixture ID list given a team id key using the dict created above. N.B./ the number of games over which the past data is averaged. A large number will smooth out past performance where as a small number will result in the prediction being heavily reliant on very recent form. This is worth testing the ml model build phase.

#5 game sliding average.
df_ml_5 = average_stats_df(5, team_list, team_fixture_id_dict, game_stats)

#10 game sliding average.
df_ml_10 = average_stats_df(10, team_list, team_fixture_id_dict, game_stats)
    
    
#creating and saving the ml dataframe with a 5 game sliding average.
df_for_ml_5_v2 = creating_ml_df(df_ml_5)
with open('2019_prem_generated_clean/2019_prem_df_for_ml_5_v2.txt', 'wb') as myFile:
    pickle.dump(df_for_ml_5_v2, myFile)

#creating and saving the ml dataframe with a 10 game sliding average.
df_for_ml_10_v2 = creating_ml_df(df_ml_10)
with open('2019_prem_generated_clean/2019_prem_df_for_ml_10_v2.txt', 'wb') as myFile:
    pickle.dump(df_for_ml_10_v2, myFile)


# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')



