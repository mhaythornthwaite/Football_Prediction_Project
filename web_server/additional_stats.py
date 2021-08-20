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

stats_dict_saved_name = '2019_2020_2021_prem_all_stats_dict.txt'
fixtures_saved_name = '2019_2020_2021_premier_league_fixtures_df.csv'
results_dict_saved_name = '2019_2020_2021_additional_stats_dict.txt'


#------------------------------ ADDITIONAL STATS ------------------------------

#in this section we will load our already generated stats dictionary and apply some slight transforms to get a df per team which has the past results and the teams played. This will then be used in the 'more information' dropdown / collapsible on our website


#---------- LOADING DATA ----------

with open(f'../prem_clean_fixtures_and_dataframes/{stats_dict_saved_name}', 'rb') as myFile:
    game_stats = pickle.load(myFile)
    
fixtures_clean = pd.read_csv(f'../prem_clean_fixtures_and_dataframes/{fixtures_saved_name}')


#---------- STATS DICT MANIPULATION ----------

teams_df = fixtures_clean.drop_duplicates(subset=['Home Team'])
teams_df = teams_df.drop(['Fixture ID', 'Game Date', 'Away Team', 'Away Team ID', 'Home Team Goals', 'Away Team Goals', 'Away Team Logo'], axis=1)
teams_df = teams_df.sort_values(by=['Home Team ID'])
teams_df = teams_df.reset_index(drop=True)
teams_df = teams_df.rename(columns={'Home Team ID': 'Team ID', 'Home Team': 'Team', 'Home Team Logo': 'Team Logo'})

def team_data(teams_df, team_id, return_data):
    '''
    return_data can be specified as one of the following three variables: 'Venue', 'Team', 'Team Logo'
    '''
    team = teams_df.loc[teams_df['Team ID'] == team_id]
    item = team[return_data]
    item = item.to_string(index=False)
    return item

test = team_data(teams_df, 50, 'Team')


#---------- DF MANIPULATION ----------

#in this section we will create a df for each team, containing all the basic information on all past games. This can then be used as a display in the web application. This will then be placed into a dictionary, with the key being the team ID.

#instantiating dictionary and team ID's
results_dict = {}
teams = teams_df['Team ID']

for team in teams:
    
    df = pd.DataFrame(columns=['Fixture_ID', 'Date', 'Home_Team_ID','Away_Team_ID','Home_Team','Away_Team','Home_Team_Score','Away_Team_Score','Result','Home_Team_Logo','Away_Team_Logo'])
    
    dic = game_stats[team]
    fixture_id = list(dic.keys())
    
    if len(dic) == 0:
        nan_df = results_dict[33]
        nan_df['Home_Team'] = 'N/A'
        nan_df['Away_Team'] = 'N/A'
        nan_df['Home_Team_Score'] = 0
        nan_df['Away_Team_Score'] = 0
        nan_df['Fixture_ID'] = 'N/A'
        #nan_df['Date'] = 'N/A'
        nan_df['Home_Team_ID'] = 'N/A'
        nan_df['Away_Team_ID'] = 'N/A'
        nan_df['Home_Team_Logo'] = 'N/A'
        nan_df['Away_Team_Logo'] = 'N/A'
        nan_df['Result'] = 'N/A'
        results_dict[team] = nan_df
        continue
    
    game = dic[fixture_id[0]]
    
    date = []
    home_team_id = []
    away_team_id = []
    home_team = []
    away_team = []
    home_team_score = []
    away_team_score = []
    home_team_logo = []
    away_team_logo = []
    results = []
    
    
    for i, fix_id in enumerate(fixture_id):
        game = dic[fix_id]
        
        date.append(game['Game Date'].iloc[0])
        home_team_id.append(game['Team ID'].iloc[0])
        away_team_id.append(game['Team ID'].iloc[1])
        home_team_score.append(game['Goals'].iloc[0])
        away_team_score.append(game['Goals'].iloc[1])
        
        
    df['Fixture_ID'] = fixture_id
    df['Date'] = date
    df['Home_Team_ID'] = home_team_id
    df['Away_Team_ID'] = away_team_id
    df['Home_Team_Score'] = home_team_score
    df['Away_Team_Score'] = away_team_score
        
    
    for i, home_team_ID in enumerate(df['Home_Team_ID']):
        home_team_str = team_data(teams_df, home_team_ID, 'Team')
        home_team_logo_str = team_data(teams_df, home_team_ID, 'Team Logo')
        home_team.append(home_team_str)
        home_team_logo.append(home_team_logo_str)
    
    for i, away_team_ID in enumerate(df['Away_Team_ID']):
        away_team_str = team_data(teams_df, away_team_ID, 'Team')
        away_team_logo_str = team_data(teams_df, away_team_ID, 'Team Logo')
        away_team.append(away_team_str)
        away_team_logo.append(away_team_logo_str)
    
    
    df['Home_Team'] = home_team
    df['Away_Team'] = away_team
    df['Home_Team_Logo'] = home_team_logo
    df['Away_Team_Logo'] = away_team_logo
    
    df = df.sort_values(by='Date', ascending=False)
    df = df.reset_index(drop=True)
    
    
    for i, home_team_ID in enumerate(df['Home_Team_ID']):
        
        if home_team_ID == team:
            home = True
        else:
            home = False
        
        home_score = df['Home_Team_Score'].iloc[i]
        away_score = df['Away_Team_Score'].iloc[i]
        
        if home:
            if home_score > away_score:
                result = 'W'
            elif home_score == away_score:
                result = 'D'
            elif home_score < away_score:
                result = 'L'
        
        if home==False:
            if home_score < away_score:
                result = 'W'
            elif home_score == away_score:
                result = 'D'
            elif home_score > away_score:
                result = 'L'        
            
        results.append(result)
        
    df['Result'] = results
      
    results_dict[team] = df


    
with open(f'../prem_clean_fixtures_and_dataframes/{results_dict_saved_name}', 'wb') as myFile:
    pickle.dump(results_dict, myFile)


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
