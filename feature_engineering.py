# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:23:13 2020

@author: mhayt
"""


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pandas as pd
import math
import pickle


#----------------------------- FEATURE ENGINEERING ----------------------------


with open('2019_prem_generated_clean/2019_prem_all_stats_dict.txt', 'rb') as myFile:
    game_stats = pickle.load(myFile)

df_ready_for_ml = pd.DataFrame({})

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
    sub_dict = {team:fix_id_list}
    team_fixture_id_dict.update(sub_dict)

#the list of fixtures was first home then away, we want them in chronological order so we need to sort them.
for team in team_fixture_id_dict:
    team_fixture_id_dict[team].sort()


#we can now iterate over the fixture ID list given a team id key using the dict created above

#creating the features
team_total_shots = []
team_shots_inside_box = []     
team_fouls = []
team_corners = []
team_posession = []
team_pass_accuracy = []
team_goals = []

opponent_total_shots = []
opponent_shots_inside_box = []     
opponent_fouls = []
opponent_corners = []
opponent_posession = []
opponent_pass_accuracy = []
opponent_goals = []


for team_id in team_list[:1]:
    team = game_stats[team_id] #team dictionary
    for game_id in team_fixture_id_dict[team_id]:
        game = team[game_id] #game df
        temp_index = pd.Index(game['Team Identifier'])
        team_ind = temp_index.get_loc(1)
        opponent_ind = temp_index.get_loc(2)
        
        #team features
        team_total_shots.append(game['Total Shots'][team_ind])
        team_shots_inside_box.append(game['Shots insidebox'][team_ind])
        team_fouls.append(game['Fouls'][team_ind])
        team_corners.append(game['Corner Kicks'][team_ind])
        team_posession.append(game['Ball Possession'][team_ind])
        team_pass_accuracy.append(game['Passes %'][team_ind])
        team_goals.append(game['Goals'][team_ind])     
        
        #opponent features
        opponent_total_shots.append(game['Total Shots'][opponent_ind])
        opponent_shots_inside_box.append(game['Shots insidebox'][opponent_ind])
        opponent_fouls.append(game['Fouls'][opponent_ind])
        opponent_corners.append(game['Corner Kicks'][opponent_ind])
        opponent_posession.append(game['Ball Possession'][opponent_ind])
        opponent_pass_accuracy.append(game['Passes %'][opponent_ind])
        opponent_goals.append(game['Goals'][opponent_ind])  
    
        #current test code - delete upon completion
        #print(game['Shots on Goal'][team_ind])
        #print(game['Shots on Goal'])
        #print(game_id)
        #print(team_id)
        #df_ready_for_ml['Average team shots'] = 
        #df_ready_for_ml['Average oponent shots']
    pass


sliding_game_index = [1,2,3,4,5,6,7,8,9]   
    



# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')