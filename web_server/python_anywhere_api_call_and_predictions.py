# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:22:55 2020

@author: mhayt
"""

#-------------------------------- API-FOOTBALL --------------------------------

#The website supporting this project is currently being hosted on pythonanywhere. Currently only a single script may be scheduled and therefore we need to commbine multiple scripts which generate API calls, and re-generate the predictions based on upcoming games. We also need to adhere to the absolute paths requirired by pythonanywhere. This script will result in automatic updates of the predictions website every midnight.

import requests
import pandas as pd
import math
from os import listdir


#------------------------------- INPUT VARIABLES ------------------------------

#Please state the year of investigation.

YEAR = 2020
YEAR_str = str(YEAR)

request_league_ids = False
request_fixtures = True
request_missing_game_stats = True


#------------------------------ REQUEST FUNCTIONS -----------------------------


api_key = ''

def get_api_data(base_url, end_url):
    url = base_url + end_url
    headers = {'X-RapidAPI-Key': api_key}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f'error {res.status_code}')
    res_t = res.text
    return res_t


def slice_api(api_str_output, start_char, end_char):
  e = len(api_str_output) - end_char
  s = start_char
  output = api_str_output[s:e]
  return output


def save_api_output(save_name, jason_data, json_data_path=''):
    writeFile = open(json_data_path + save_name + '.json', 'w')
    writeFile.write(jason_data)
    writeFile.close()


def read_json_as_pd_df(json_data, json_data_path='', orient_def='records'):
    output = pd.read_json(json_data_path + json_data, orient=orient_def)
    return output



#---------------------------- REQUESTING BASIC DATA ---------------------------

base_url = 'https://v2.api-football.com/'

def req_prem_fixtures_id(season_code, year=YEAR_str):
    #request to the api for the data
    premier_league_fixtures_raw = get_api_data(base_url, f'/fixtures/league/{season_code}/')

    #cleaning the data in preparation for loading into a dataframe
    premier_league_fixtures_sliced = slice_api(premier_league_fixtures_raw, 33, 2)

    #saving the clean data as a json file
    save_api_output(f'/home/matthaythornthwaite/Football_Prediction_Project/{year}_premier_league_fixtures', premier_league_fixtures_sliced, json_data_path = '/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/')

    #loading the json file as a DataFrame
    premier_league_fixtures_df = read_json_as_pd_df(f'/home/matthaythornthwaite/Football_Prediction_Project/{year}_premier_league_fixtures.json', json_data_path='/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/')
    return premier_league_fixtures_df


#requesting data on the premier leagues, we will use this response to get the league_id of the season we were interested in
if request_league_ids:
    leagues = premier_league_fixtures_raw = get_api_data(base_url, 'leagues/search/premier_league')

if YEAR == 2019:
    season_id = 524
elif YEAR == 2020:
    season_id = 2790
else:
    print('please lookup season id and specify this as season_id variable')

#requesting the fixture list using the function req_prem_fixture_id
if request_fixtures:
    fixtures = req_prem_fixtures_id(season_id, YEAR_str)


def load_prem_fixtures_id(year=YEAR_str):
    premier_league_fixtures_df = read_json_as_pd_df(f'{year}_premier_league_fixtures.json', json_data_path='/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/')
    return premier_league_fixtures_df


fixtures = load_prem_fixtures_id()


#------------------------- MAKING CLEAN FIXTURE LIST --------------------------

fixtures = pd.read_json(f'/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/{YEAR_str}_premier_league_fixtures.json', orient='records')

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

for i in fixtures.index:
    x = str(fixtures['homeTeam'][i]['team_name'])
    fixtures.at[i, 'Home Team'] = x

for i in fixtures.index:
    x = str(fixtures['awayTeam'][i]['team_name'])
    fixtures.at[i, 'Away Team'] = x

for i in fixtures.index:
    x = str(fixtures['homeTeam'][i]['logo'])
    fixtures.at[i, 'Home Team Logo'] = x

for i in fixtures.index:
    x = str(fixtures['awayTeam'][i]['logo'])
    fixtures.at[i, 'Away Team Logo'] = x


fixtures_clean = pd.DataFrame({'Fixture ID': fixtures['fixture_id'], 'Game Date': fixtures['Game Date'], 'Home Team ID': fixtures['HomeTeamID'], 'Away Team ID': fixtures['AwayTeamID'], 'Home Team Goals': fixtures['goalsHomeTeam'], 'Away Team Goals': fixtures['goalsAwayTeam'], 'Venue': fixtures['venue'], 'Home Team': fixtures['Home Team'], 'Away Team': fixtures['Away Team'], 'Home Team Logo': fixtures['Home Team Logo'], 'Away Team Logo': fixtures['Away Team Logo']})

fixtures_clean.to_csv(f'/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/{YEAR_str}_premier_league_fixtures_df.csv', index=False)


#------------------------- STITCHINING CLEAN FIXTURE LIST --------------------------

#in this section we simply load the 2019 fixtures and the 2020 fixtures and stitch the two dataframes together.

fixtures_clean_2019 = pd.read_csv(f'/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/2019_premier_league_fixtures_df.csv')

fixtures_clean_2020 = pd.read_csv(f'/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/2020_premier_league_fixtures_df.csv')

fixtures_clean_combined = pd.concat([fixtures_clean_2019, fixtures_clean_2020])
fixtures_clean_combined = fixtures_clean_combined.reset_index(drop=True)

fixtures_clean_combined.to_csv(f'/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/2019_2020_premier_league_fixtures_df.csv', index=False)


#----- AUTOMATING MISSING DATA COLLECTION -----

#in this section we will search through our exisiting database (prem_game_stats_json_files folder) and request the game data of any missing games that have been played since we last requested data.


#listing the json data already collected
existing_data_raw = listdir('/home/matthaythornthwaite/Football_Prediction_Project/prem_game_stats_json_files/')

#removing '.json' from the end of this list
existing_data = []
for i in existing_data_raw:
    existing_data.append(int(i[:-5]))

#creating a list with the missing
missing_data = []
for i in fixtures_clean.index:
    fix_id = fixtures_clean['Fixture ID'].iloc[i]
    if fix_id not in existing_data:
        if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
            missing_data.append(fix_id)


def req_prem_stats_list(missing_data):
    if len(missing_data) > 100:
        print('This request exceeds 100 request limit and has not been completed')
    else:
        if len(missing_data) > 0:
            print('Data collected for the following fixtures:')
        for i in missing_data:
            print(i)
            fix_id = str(i)
            fixture_raw = get_api_data(base_url, '/statistics/fixture/' + fix_id + '/')
            fixture_sliced = slice_api(fixture_raw, 34, 2)
            save_api_output('/home/matthaythornthwaite/Football_Prediction_Project/prem_game_stats_json_files/' + fix_id, fixture_sliced)

if request_missing_game_stats:
    req_prem_stats_list(missing_data)



#--------------------------------- PREDICTIONS --------------------------------

import pickle
import numpy as np
from ml_functions.feature_engineering_functions import average_stats_df, mod_df


#------------------------------- INPUT VARIABLES ------------------------------

fixtures_saved_name = '2019_2020_premier_league_fixtures_df.csv'

stats_dict_saved_name = '2019_2020_prem_all_stats_dict.txt'

df_10_saved_name = '2019_2020_prem_df_for_ml_10_v2.txt'

path_to_model = '/home/matthaythornthwaite/Football_Prediction_Project/ml_model_build_random_forest/ml_models/random_forest_model_10.pk1'


#----------------------------- FEATURE ENGINEERING ----------------------------

with open(f'/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/{stats_dict_saved_name}', 'rb') as myFile:
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

df_10_upcom_fix_e = average_stats_df(10, team_list, team_fixture_id_dict_reduced, game_stats, making_predictions=True)
df_10_upcom_fix = mod_df(df_10_upcom_fix_e, making_predictions=True)

#loading fixtures dataframe, we will work with the clean version but it is good to be aware of what is available in the raw version.
fixtures_clean = pd.read_csv(f'/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/{fixtures_saved_name}')

#creating a df with unplayed games only
played_games = []
for i in range(0, len(fixtures_clean)):
    if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
        played_games.append(i)

unplayed_games = fixtures_clean.drop(fixtures_clean.index[played_games])
unplayed_games = unplayed_games.reset_index(drop=True)
unplayed_games = unplayed_games.drop(['Home Team Goals', 'Away Team Goals'], axis=1)

#loading df for the labels
with open(f'/home/matthaythornthwaite/Football_Prediction_Project/prem_clean_fixtures_and_dataframes/{df_10_saved_name}', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

column_list = df_ml_10.columns.tolist()

#instatiating the df for predictions with zeros
df_for_predictions = pd.DataFrame(np.zeros((68, 14)))
df_for_predictions.columns = column_list[:14]

#adding the home and away team id
df_for_predictions = pd.DataFrame(np.zeros((len(unplayed_games), 14)))
df_for_predictions.columns = column_list[:14]
df_for_predictions['Home Team ID'] = unplayed_games['Home Team ID']
df_for_predictions['Away Team ID'] = unplayed_games['Away Team ID']
df_for_predictions['Home Team'] = unplayed_games['Home Team']
df_for_predictions['Away Team'] = unplayed_games['Away Team']
df_for_predictions['Game Date'] = unplayed_games['Game Date']


# ---------- MODELLING MISSING GAME DATA ----------
#if our newly promoted team has not yet played 10 games we need to fill in this gap in order to make a prediction. Lets take the 3 relegated teams, avergae these and use that for all newly promoted teams. 

relegated_id_1 = 35
relegated_id_2 = 38
relegated_id_3 = 71

rel_1_df = (df_10_upcom_fix.loc[df_10_upcom_fix['Team ID'] == relegated_id_1]).reset_index(drop=True)
rel_2_df = (df_10_upcom_fix.loc[df_10_upcom_fix['Team ID'] == relegated_id_2]).reset_index(drop=True)
rel_3_df = (df_10_upcom_fix.loc[df_10_upcom_fix['Team ID'] == relegated_id_3]).reset_index(drop=True)

average_df = rel_1_df.add(rel_2_df, fill_value=0)
average_df = average_df.add(rel_3_df, fill_value=0)
average_df = average_df.div(3)


# ---------- POPULATING 'df_for_predictions' WITH STATS ----------

for i in range(0, len(unplayed_games)):
    #getting home team id and index
    home_team = unplayed_games['Home Team ID'].iloc[i]
    home_team_index = df_10_upcom_fix[df_10_upcom_fix['Team ID']==home_team].index.values

    #getting away team id and index
    away_team = unplayed_games['Away Team ID'].iloc[i]
    away_team_index = df_10_upcom_fix[df_10_upcom_fix['Team ID']==away_team].index.values

    #getting the home and away team stats given the index of the teams. This still a df. To replace in the df_for_predictions we need this to be a list. This turns out to be quite complex (steps 2 through to 5).
    #if the team is newly promoted they will not have any stats in df_10_upcom_fix. If this is the case we need to replace the missing data with modelled data
    team_ids = list(df_10_upcom_fix['Team ID'])

    if home_team in team_ids:
        h1 = df_10_upcom_fix.iloc[home_team_index]
    else:
        h1 = average_df

    if away_team in team_ids:
        a1 = df_10_upcom_fix.iloc[away_team_index]
    else:
        a1 = average_df

    h2 = h1.T
    a2 = a1.T

    h3 = h2.values.tolist()
    a3 = a2.values.tolist()

    h4 = []
    for j in range(0, len(h3)):
        h4.append(h3[j][0])

    a4 = []
    for k in range(0, len(a3)):
        a4.append(a3[k][0])

    h5 = h4[0:7]
    a5 = a4[0:7]

    df_for_predictions.iloc[i, 0:7] = h5
    df_for_predictions.iloc[i, 7:14] = a5


#--------------------------- MAKING THE PREDICTIONS ---------------------------

clf = pickle.load(open(f'{path_to_model}', 'rb'))

df_for_predictions_r = df_for_predictions.drop(['Home Team ID', 'Away Team ID', 'Home Team', 'Away Team', 'Game Date'], axis=1)

predictions_raw = clf.predict_proba(df_for_predictions_r)

predictions_df = pd.DataFrame(data=predictions_raw, index=range(0, len(predictions_raw)), columns=['Away Win', 'Draw', 'Home Win'])

predictions_df[predictions_df.select_dtypes(include=['number']).columns] *= 100
predictions_df = predictions_df.round(1)

predictions = pd.concat([unplayed_games, predictions_df], axis=1, join='inner')

re_order_cols = ['Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win', 'Game Date', 'Venue', 'Home Team Logo', 'Away Team Logo', 'Home Team ID', 'Away Team ID', 'Fixture ID', 'index']

predictions = predictions.reindex(columns=re_order_cols)

with open('/home/matthaythornthwaite/Football_Prediction_Project/predictions/pl_predictions.csv', 'wb') as myFile:
    pickle.dump(predictions, myFile)
with open('/home/matthaythornthwaite/Football_Prediction_Project/web_server/pl_predictions.csv', 'wb') as myFile:
    pickle.dump(predictions, myFile)


# ----------------------------------- END -------------------------------------
