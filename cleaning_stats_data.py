# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:31:13 2020

@author: mhayt
"""


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pandas as pd
import math
import pickle


#---------------------------- CREATING DF PER TEAM ----------------------------
#in this section we will create a nested dictionary containing the 20 teams, each with a value as another dictionary. In this dictionary we will have the game id along with the game dataframe.

fixtures_clean = pd.read_csv('prem_clean_fixtures_and_dataframes/2019_premier_league_fixtures_df.csv')

#creating the 'fixtures_clean' ID index which we will use to take data from this dataframe and add to each of our individual fixture stats dataframe.
fixtures_clean_ID_index = pd.Index(fixtures_clean['Fixture ID'])

#team id list that we can iterate over
team_id_list = (fixtures_clean['Home Team ID'].unique()).tolist()

#creating our dictionary which we will populate with data
all_stats_dict = {}

#nested for loop to create nested dictionary, first key by team id, second key by fixture id.
for team in team_id_list:
    
    #working the home teams
    team_fixture_list = []
    for i in fixtures_clean.index[:]:
        if fixtures_clean['Home Team ID'].iloc[i] == team:
            if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    all_stats_dict[team] = {}
    for j in team_fixture_list:
        #loading df
        df = pd.read_json('2019_prem_game_stats/' + str(j) + '.json', orient='values')
        #removing percentage symbol in possession and passes and conv to int
        df['Ball Possession'] = df['Ball Possession'].str.replace('[\%]', '').astype(int)
        df['Passes %'] = df['Passes %'].str.replace('[\%]', '').astype(int)
        #adding home vs away goals to df
        temp_index = fixtures_clean_ID_index.get_loc(j)
        home_goals = fixtures_clean['Home Team Goals'].iloc[temp_index]
        away_goals = fixtures_clean['Away Team Goals'].iloc[temp_index]
        df['Goals'] = [home_goals, away_goals]
        #adding points data
        if home_goals > away_goals:
            df['Points'] = [2,0]
        elif home_goals == away_goals:
            df['Points'] = [1,1]
        elif home_goals < away_goals:
            df['Points'] = [0,2]
        else:
            df['Points'] = ['nan', 'nan']
        #adding home-away identifier to df
        df['Team Identifier'] = [1,2]
        #adding team id
        df['Team ID'] = [team, fixtures_clean['Away Team ID'].iloc[temp_index]]
        #adding game date
        gd = fixtures_clean['Game Date'].iloc[temp_index]
        df['Game Date'] = [gd, gd]
        #adding this modified df to nested dictionary
        sub_dict_1 = {j:df}
        all_stats_dict[team].update(sub_dict_1)
        
    #working the away teams    
    team_fixture_list = []    
    for i in fixtures_clean.index[:]:
        if fixtures_clean['Away Team ID'].iloc[i] == team:
            if math.isnan(fixtures_clean['Away Team Goals'].iloc[i]) == False:
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    for j in team_fixture_list:
        #loading df
        df = pd.read_json('2019_prem_game_stats/' + str(j) + '.json', orient='values')
        #removing percentage symbol in possession and passes and conv to int
        df['Ball Possession'] = df['Ball Possession'].str.replace('[\%]', '').astype(int)
        df['Passes %'] = df['Passes %'].str.replace('[\%]', '').astype(int)
        #adding home vs away goals to df
        temp_index = fixtures_clean_ID_index.get_loc(j)
        home_goals = fixtures_clean['Home Team Goals'].iloc[temp_index]
        away_goals = fixtures_clean['Away Team Goals'].iloc[temp_index]
        df['Goals'] = [home_goals, away_goals]
        #adding points data
        if home_goals > away_goals:
            df['Points'] = [2,0]
        elif home_goals == away_goals:
            df['Points'] = [1,1]
        elif home_goals < away_goals:
            df['Points'] = [0,2]
        else:
            df['Points'] = ['nan', 'nan']
        #adding home-away identifier to df
        df['Team Identifier'] = [2,1]       
        #adding team id
        df['Team ID'] = [fixtures_clean['Home Team ID'].iloc[temp_index], team]
        #adding game date
        gd = fixtures_clean['Game Date'].iloc[temp_index]
        df['Game Date'] = [gd, gd]
        #adding this modified df to nested dictionary
        sub_dict_1 = {j:df}
        all_stats_dict[team].update(sub_dict_1)
        

#saving our generated dictionary as a pickle file to import into a later python file.

with open('prem_clean_fixtures_and_dataframes/2019_prem_all_stats_dict.txt', 'wb') as myFile:
    pickle.dump(all_stats_dict, myFile)

with open('prem_clean_fixtures_and_dataframes/2019_prem_all_stats_dict.txt', 'rb') as myFile:
    loaded_dict_test = pickle.load(myFile)


# -------------------------------- NEW SEASON ---------------------------------

#in this section we need to populate our nested dictionary with pseudo game data for the teams that have been newly promoted. The previous ten games from the 3 relegated, totaling 30 fixtures will be averaged to create a single game stats df. This will be replicated 10 times for each newly promoted team. As the early season progresses, genuine game data will be used instead for the predictions.

#specify the id of the relegated teams
relegated_id_1 = 35
relegated_id_2 = 38
relegated_id_3 = 71

#listing the last 10 fixtures of each relegated team.
relegated_id_1_fixtures = ([i for i in all_stats_dict[relegated_id_1]])
relegated_id_1_fixtures.sort()
relegated_id_1_fixtures = relegated_id_1_fixtures[-10:]

relegated_id_2_fixtures = ([i for i in all_stats_dict[relegated_id_2]])
relegated_id_2_fixtures.sort()
relegated_id_2_fixtures = relegated_id_2_fixtures[-10:]

relegated_id_3_fixtures = ([i for i in all_stats_dict[relegated_id_3]])
relegated_id_3_fixtures.sort()
relegated_id_3_fixtures = relegated_id_3_fixtures[-10:]


#instantiating the 'average_df' which will be used as the base in the following iterations.
average_df = all_stats_dict[relegated_id_1][(relegated_id_1_fixtures[0])]
average_df = average_df.set_index('Team Identifier')


#iterating over each of the last 10 games each relegated team played. Each df is added to the average_df.
for i in relegated_id_1_fixtures[1:]:
    df = all_stats_dict[relegated_id_1][i]
    df = df.set_index('Team Identifier')
    average_df = average_df.add(df, fill_value=0)

for i in relegated_id_2_fixtures:
    df = all_stats_dict[relegated_id_2][i]
    df = df.set_index('Team Identifier')
    average_df = average_df.add(df, fill_value=0)
    
for i in relegated_id_3_fixtures:
    df = all_stats_dict[relegated_id_3][i]
    df = df.set_index('Team Identifier')
    average_df = average_df.add(df, fill_value=0)


#because we summed all 30 df's, we now need to divide by 30 to get an average df 
average_df = average_df.drop('Game Date', axis=1)
average_df = average_df.div(30)



# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')