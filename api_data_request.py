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
from os import listdir

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


fixtures = req_prem_fixtures_id()
    


def load_prem_fixtures_id():
    premier_league_fixtures_df = read_json_as_pd_df('2019_premier_league_fixtures.json', json_data_path='2019_prem_generated_clean/')
    return premier_league_fixtures_df

fixtures = load_prem_fixtures_id()



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

fixtures_clean.to_csv('2019_prem_generated_clean/2019_premier_league_fixtures_df.csv', index=False)




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
 

#----- AUTOMATING MISSING DATA COLLECTION -----

#in this section we will search through our exisiting database (2019_prem_game_stats folder) and request the game data of any missing games that have been played since we last requested data.


#listing the json data already collected
existing_data_raw = listdir('2019_prem_game_stats/')

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
            save_api_output('2019_prem_game_stats/' + fix_id, fixture_sliced)


req_prem_stats_list(missing_data)



# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')