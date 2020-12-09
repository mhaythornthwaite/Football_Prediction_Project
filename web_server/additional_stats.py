# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:37:32 2020

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import time
start=time.time()

import pickle
import pandas as pd


#------------------------------- INPUT VARIABLES ------------------------------

stats_dict_saved_name = '2019_2020_prem_all_stats_dict.txt'


#------------------------------ ADDITIONAL STATS ------------------------------

#in this section we will load our already generated stats dictionary and apply some slight transforms to get a df per team which has the past results and the teams played. This will then be used in the 'more information' dropdown / collapsible on our website

with open(f'../prem_clean_fixtures_and_dataframes/{stats_dict_saved_name}', 'rb') as myFile:
    game_stats = pickle.load(myFile)

df = pd.DataFrame(columns=['Fixture_ID', 'Date', 'Home_Team_ID','Away_Team_ID','Home_Team','Away_Team','Home_Team_Score','Away_Team_Score'])

dic = game_stats[33]
fixture_id = list(dic.keys())

game = dic[fixture_id[1]]


date = []
home_team_id = []
away_team_id = []
home_team = []
away_team = []
home_team_score = []
away_team_score = []


for i, fix_id in enumerate(fixture_id):
    game = dic[fix_id]
    
    date.append(game['Game Date'].iloc[0])
    home_team_id.append(game['Team ID'].iloc[0])
    away_team_id.append(game['Team ID'].iloc[1])
    
df['Fixture_ID'] = fixture_id
df['Date'] = date
df['Home_Team_ID'] = home_team_id
df['Away_Team_ID'] = away_team_id

df2 = df.sort_values(by='Date')


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
