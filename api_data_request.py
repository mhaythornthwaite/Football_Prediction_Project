# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:33:10 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import requests
import pandas as pd
import math


#IMPORTANT NOTE./ in this documentation all the examples are realized with the url provided for rapidApi, if you have subscribed directly with us you will have to replace https://api-football-v1.p.rapidapi.com/v2/ by https://v2.api-football.com/


#------------------------------ REQUEST FUNCTIONS -----------------------------


def get_api_data(base_url, end_url):
    url = base_url + end_url
    headers = {'X-RapidAPI-Key':'f6d8a1ef214463be7b6afa8fc8054b5b'}
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


def save_api_output(save_name, jason_data):
    writeFile = open(save_name + '.json', 'w')
    writeFile.write(jason_data)
    writeFile.close()
    
    
def read_json_as_pd_df(json_data, json_data_path='', orient_def='records'):
    output = pd.read_json(json_data_path + json_data, orient=orient_def)
    return output
  
    

#---------------------------- REQUESTING BASIC DATA ---------------------------

base_url = 'https://v2.api-football.com/'

def req_league_id():
    #request to the api for the data
    league_id_data_raw = get_api_data(base_url, 'leagues')

    #cleaning the data in preparation for loading into a dataframe
    league_id_data_sliced = slice_api(league_id_data_raw, 33, 2)

    #saving the clean data as a json file
    save_api_output('league_id_data', league_id_data_sliced)

    #loading the json file as a DataFrame - premier league id: 524
    league_id_data_df = read_json_as_pd_df('league_id_data.json')
    return league_id_data_df
    
def req_prem_teams_id():
    #request to the api for the data
    premier_league_teams_raw = get_api_data(base_url, '/teams/league/524/')

    #cleaning the data in preparation for loading into a dataframe
    premier_league_teams_sliced = slice_api(premier_league_teams_raw, 29, 2)

    #saving the clean data as a json file
    save_api_output('premier_league_teams', premier_league_teams_sliced)

    #loading the json file as a DataFrame
    premier_league_teams_df = read_json_as_pd_df('premier_league_teams.json')
    return premier_league_teams_df

def req_prem_fixtures_id():
    #request to the api for the data
    premier_league_fixtures_raw = get_api_data(base_url, '/fixtures/league/524/')

    #cleaning the data in preparation for loading into a dataframe
    premier_league_fixtures_sliced = slice_api(premier_league_fixtures_raw, 33, 2)

    #saving the clean data as a json file
    save_api_output('premier_league_fixtures', premier_league_fixtures_sliced)

    #loading the json file as a DataFrame
    premier_league_fixtures_df = read_json_as_pd_df('premier_league_fixtures.json')
    return premier_league_fixtures_df

def req_burnley_match_stats(save_path):
    #request to the api for the data
    burnley_fixture_raw = get_api_data(base_url, '/statistics/fixture/157018/')

    #cleaning the data in preparation for loading into a dataframe
    burnley_fixture_sliced = slice_api(burnley_fixture_raw, 34, 2)

    #saving the clean data as a json file
    save_api_output(save_path + 'burnley_fixture', burnley_fixture_sliced)

    #loading the json file as a DataFrame
    burnley_fixture_df = read_json_as_pd_df(save_path + 'burnley_fixture.json', orient_def='values')
    return burnley_fixture_df


#burnley = req_burnley_match_stats()
#fixtures = req_prem_fixtures_id()
    

def load_league_id():
    league_id_data_df = read_json_as_pd_df('league_id_data.json')
    return league_id_data_df

def load_prem_teams_id():
    premier_league_teams_df = read_json_as_pd_df('premier_league_teams.json')
    return premier_league_teams_df

def load_prem_fixtures_id():
    premier_league_fixtures_df = read_json_as_pd_df('premier_league_fixtures.json')
    return premier_league_fixtures_df

def load_burnley_match_stats():
    burnley_fixture_df = read_json_as_pd_df('burnley_fixture.json', orient_def='values')
    return burnley_fixture_df



#-------------------------- REQUESTING SPECIFIC STATS -------------------------


fixtures = load_prem_fixtures_id()
burnley = load_burnley_match_stats()

print(fixtures.info())

print(fixtures['homeTeam'])

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



#creating clean past fixture list DataFrame       

fixtures_clean = pd.DataFrame({'Fixture ID': fixtures['fixture_id'], 'Game Date': fixtures['Game Date'], 'Home Team ID': fixtures['HomeTeamID'], 'Away Team ID': fixtures['AwayTeamID'], 'Home Team Goals': fixtures['goalsHomeTeam'], 'Away Team Goals': fixtures['goalsAwayTeam']})

      
    
def req_prem_stats(start_index, end_index):
    for i in fixtures_clean.index[start_index:end_index]:
        if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
            fix_id = str(fixtures_clean['Fixture ID'].iloc[i])
            fixture_raw = get_api_data(base_url, '/statistics/fixture/' + fix_id + '/')
            fixture_sliced = slice_api(fixture_raw, 34, 2)
            save_api_output('2019_prem_game_stats/' + fix_id, fixture_sliced)
        
#req_prem_stats(50, 70)
 







# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')