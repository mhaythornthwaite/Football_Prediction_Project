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


def load_burnley_match_stats():
    burnley_fixture_df = read_json_as_pd_df('burnley_fixture.json', orient_def='values')
    return burnley_fixture_df


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
#in this section we will create a dictionary containing the dataframes of each team.

list_44 = []

for i in fixtures_clean.index:
    if fixtures_clean['Home Team ID'].iloc[i] == 44:
        if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
            list_44.append(fixtures_clean['Fixture ID'].iloc[i])
        
print(list_44)

li_44 = list_44[:3]

test_load = pd.read_json('2019_prem_game_stats/157018.json', orient='values')

df_dict = {'test': test_load}       

test = df_dict['test']['Shots on Goal'].iloc[1]


#for i in li_44:
 #   i = read_json_as_pd_df('burnley_fixture.json', orient_def='values')






# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')