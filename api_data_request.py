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


def save_api_output(save_name, jason_data, json_data_path=''):
    writeFile = open(json_data_path + save_name + '.json', 'w')
    writeFile.write(jason_data)
    writeFile.close()
    
    
def read_json_as_pd_df(json_data, json_data_path='', orient_def='records'):
    output = pd.read_json(json_data_path + json_data, orient=orient_def)
    return output
  
    

#---------------------------- REQUESTING BASIC DATA ---------------------------

base_url = 'https://v2.api-football.com/'



def req_prem_fixtures_id():
    #request to the api for the data
    premier_league_fixtures_raw = get_api_data(base_url, '/fixtures/league/524/')

    #cleaning the data in preparation for loading into a dataframe
    premier_league_fixtures_sliced = slice_api(premier_league_fixtures_raw, 33, 2)

    #saving the clean data as a json file
    save_api_output('2019_premier_league_fixtures', premier_league_fixtures_sliced, json_data_path = '2019_prem_generated_clean/')

    #loading the json file as a DataFrame
    premier_league_fixtures_df = read_json_as_pd_df('2019_premier_league_fixtures.json', json_data_path='2019_prem_generated_clean/')
    return premier_league_fixtures_df


#fixtures = req_prem_fixtures_id()
    


def load_prem_fixtures_id():
    premier_league_fixtures_df = read_json_as_pd_df('2019_premier_league_fixtures.json', json_data_path='2019_prem_generated_clean/')
    return premier_league_fixtures_df

fixtures = load_prem_fixtures_id()


#-------------------------- REQUESTING SPECIFIC STATS -------------------------

fixtures_clean = pd.read_csv('2019_prem_generated_clean/2019_premier_league_fixtures_df.csv')
    
def req_prem_stats(start_index, end_index):
    for i in fixtures_clean.index[start_index:end_index]:
        if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
            fix_id = str(fixtures_clean['Fixture ID'].iloc[i])
            fixture_raw = get_api_data(base_url, '/statistics/fixture/' + fix_id + '/')
            fixture_sliced = slice_api(fixture_raw, 34, 2)
            save_api_output('2019_prem_game_stats/' + fix_id, fixture_sliced)
        
#req_prem_stats(288, 300)
 



# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')