# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:45:37 2023

@author: mhayt
"""

import pickle


with open('prem_clean_fixtures_and_dataframes/2019_2020_2021_2022_2023_additional_stats_dict.txt', 'rb') as myFile:
    game_stats_2023 = pickle.load(myFile)


with open('prem_clean_fixtures_and_dataframes/2019_2020_2021_2022_additional_stats_dict.txt', 'rb') as myFile:
    game_stats_2022 = pickle.load(myFile)

