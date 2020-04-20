# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:31:13 2020

@author: mhayt
"""


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import requests
import pandas as pd
import math
import pickle

#------------------------- MAKING CLEAN FIXTURE LIST --------------------------

fixtures = pd.read_json('2019_prem_generated_clean/2019_premier_league_fixtures.json', orient='records')

#creating clean past fixture list DataFrame       

for i in fixtures.index:
    x1 = str(fixtures['homeTeam'].iloc[i])[12:14]
    x = int(x1)
    fixtures.at[i, 'HomeTeamID'] = x

for i in fixtures.index:
    x1 = str(fixtures['awayTeam'].iloc[i])[12:14]
    x = int(x1)
    fixtures.at[i, 'AwayTeamID'] = x

for i in fixtures.index:
    x = str(fixtures['event_date'].iloc[i])[:10] 
    fixtures.at[i, 'Game Date'] = x

fixtures_clean = pd.DataFrame({'Fixture ID': fixtures['fixture_id'], 'Game Date': fixtures['Game Date'], 'Home Team ID': fixtures['HomeTeamID'], 'Away Team ID': fixtures['AwayTeamID'], 'Home Team Goals': fixtures['goalsHomeTeam'], 'Away Team Goals': fixtures['goalsAwayTeam']})

fixtures_clean.to_csv('2019_prem_generated_clean/2019_premier_league_fixtures_df.csv', index=False)


#---------------------------- CREATING DF PER TEAM ----------------------------
#in this section we will create a nested dictionary containing the 20 teams, each with a value as another dictionary. In this dictionary we will have the game id along with the game dataframe.


#team id list that we can iterate over
team_id_list = (fixtures_clean['Home Team ID'].unique()).tolist()

#creating our dictionary which we will populate with data
all_stats_dict = {}

#nested for loop to create nested dictionary, first key by team id, second key by fixture id.
for team in team_id_list:
    
    #working the home teams
    team_fixture_list = []
    for i in fixtures_clean.index[:50]:
        if fixtures_clean['Home Team ID'].iloc[i] == team:
            if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    all_stats_dict[team] = {}
    for j in team_fixture_list:
        df = pd.read_json('2019_prem_game_stats/' + str(j) + '.json', orient='values')
        df['Home vs Away ID'] = [1,2]
        sub_dict_1 = {j:df}
        all_stats_dict[team].update(sub_dict_1)
        
    #working the away teams    
    team_fixture_list = []    
    for i in fixtures_clean.index[:50]:
        if fixtures_clean['Away Team ID'].iloc[i] == team:
            if math.isnan(fixtures_clean['Away Team Goals'].iloc[i]) == False:
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    for j in team_fixture_list:
        df = pd.read_json('2019_prem_game_stats/' + str(j) + '.json', orient='values')
        df['Home vs Away ID'] = [2,1]
        sub_dict_1 = {j:df}
        all_stats_dict[team].update(sub_dict_1)
        

#saving our generated dictionary as a pickle file to import into a later python file.

with open('2019_prem_generated_clean/saved_all_stats_dict.txt', 'wb') as myFile:
    pickle.dump(all_stats_dict, myFile)

with open('2019_prem_generated_clean/saved_all_stats_dict.txt', 'rb') as myFile:
    loaded_dict_test = pickle.load(myFile)



# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')