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


#------------------------- MAKING CLEAN FIXTURE LIST --------------------------

fixtures = pd.read_json('prem_seasons_fixture_id/2019_premier_league_fixtures.json', orient='records')

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

fixtures_clean.to_csv('prem_seasons_fixture_id/2019_premier_league_fixtures_df.csv', index=False)


#---------------------------- CREATING DF PER TEAM ----------------------------
#in this section we will create a nested dictionary containing the 20 teams, each with a value as another dictionary. In this dictionary we will have the game id along with the game dataframe.




#team id list that we can iterate over
team_id_list = (fixtures_clean['Home Team ID'].unique()).tolist()

all_stats_dict = {}

for team in team_id_list:
    team_fixture_list = []
    for i in fixtures_clean.index[:100]:
        if fixtures_clean['Home Team ID'].iloc[i] == team:
            if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    print(team_fixture_list)
    all_stats_dict[team] = {}
    for j in team_fixture_list:
        df = pd.read_json('2019_prem_game_stats/' + str(j) + '.json', orient='values')
        df['Home vs Away ID'] = [1,2]
        sub_dict_1 = {j:df}
        all_stats_dict[team].update(sub_dict_1)
        

                

#----- TEST CODE 0 ------

a = 42
b = [1,2,3]
c = [4,5,6]
d = [7,8,9]

team_fix_list = [175423, 175424, 175425]

M_Dict = {}
for i in team_fix_list:
  M_Dict[a] = {}
  M_Dict[a][i] = b

print(M_Dict)

#----- TEST CODE -----

list_44 = []

for i in fixtures_clean.index:
    if fixtures_clean['Home Team ID'].iloc[i] == 44:
        if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
            list_44.append(fixtures_clean['Fixture ID'].iloc[i])
        
#print(list_44)

li_44 = list_44[:3]

test_load = pd.read_json('2019_prem_game_stats/157018.json', orient='values')

df_dict = {'test': test_load}       

test = df_dict['test']['Shots on Goal'].iloc[1]


#for i in li_44:
 #   i = read_json_as_pd_df('burnley_fixture.json', orient_def='values')


test_load['Home vs Away ID'] = [1,2]



# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')