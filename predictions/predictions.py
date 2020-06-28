# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:45:42 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

#!/usr/bin/python
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep)


import pandas as pd
import pickle
import numpy as np
from data_cleaning_functions.feature_engineering_functions import generate_ml_df, running_mean

#----------------------------- FEATURE ENGINEERING ----------------------------

with open('../2019_prem_generated_clean/2019_prem_all_stats_dict.txt', 'rb') as myFile:
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
    
#creating the same dictionary as above but only with the previous 10 games ready for predictions.
team_fixture_id_dict_reduced = {}
for team in team_fixture_id_dict:
    team_fixture_list_reduced = team_fixture_id_dict[team][-10:]
    sub_dict = {team:team_fixture_list_reduced}
    team_fixture_id_dict_reduced.update(sub_dict)

test = generate_ml_df(10, team_list, team_fixture_id_dict_reduced, game_stats, making_predictions=True)





    
    
# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')