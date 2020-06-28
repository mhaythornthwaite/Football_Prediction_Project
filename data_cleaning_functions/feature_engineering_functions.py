# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:13:02 2020

@author: mhayt
"""

#-------------------------------- API-FOOTBALL --------------------------------


import numpy as np
import pandas as pd
import pickle


#------------------------------- DATA PROCESSING ------------------------------


#creating a running average function
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



def generate_ml_df(games_slide, team_list, team_fixture_id_dict, game_stats, making_predictions=False):
    '''
    Output is a dataframe of averaged game stats. Included is a teams average stats over 'games_slide' number of games as well as the avergae opponent stats in those games.

    Parameters
    ----------
    games_slide : int
        Number of games to average over.
    team_list : list
        list of teams ID's. For premier league there should be 20
    team_fixture_id_dict : dict
        key: team ID, value: list of fixture ID
    game_stats : nested dict
        key: team id, second-key: fixtue ID, value: stats dataframe 
    making_predictions: bool
        default = False. Set to true if creating a prediction dataframe

    Returns
    -------
    df_ready_for_ml : dataframe
        averaged game stats

    '''
    
    if making_predictions:
        x = games_slide
        xx = 1
    else:
        x = -1
        xx = 0

  
    #creating final features which will be appended
    t_total_shots = []
    t_shots_inside_box = []     
    t_fouls = []
    t_corners = []
    t_posession = []
    t_pass_accuracy = []
    t_goals = []
    t_goals_target = []
    o_total_shots = []
    o_shots_inside_box = []     
    o_fouls = []
    o_corners = []
    o_posession = []
    o_pass_accuracy = []
    o_goals = []
    o_goals_target = []
    fix_id = []
    result_indicator = []
    o_team_ID = []
    
    for team_id in team_list[:]:
        team = game_stats[team_id] #team dictionary
        
        #creating the initial features - it is important these get overwritten with each iteration
        team_total_shots = []
        team_shots_inside_box = []     
        team_fouls = []
        team_corners = []
        team_posession = []
        team_pass_accuracy = []
        team_goals = []
        opponent_total_shots = []
        opponent_shots_inside_box = []     
        opponent_fouls = []
        opponent_corners = []
        opponent_posession = []
        opponent_pass_accuracy = []
        opponent_goals = []
        result_indicator_raw = []
        
        
        #iterating over the fixture id to create feature lists
        for game_id in team_fixture_id_dict[team_id]:
            game = team[game_id] #game df
            temp_index = pd.Index(game['Team Identifier'])
            team_ind = temp_index.get_loc(1)
            opponent_ind = temp_index.get_loc(2)
            
            #team and opponent pseudo features: list of raw feature data for each game
            team_total_shots.append(game['Total Shots'][team_ind])
            team_shots_inside_box.append(game['Shots insidebox'][team_ind])
            team_fouls.append(game['Fouls'][team_ind])
            team_corners.append(game['Corner Kicks'][team_ind])
            team_posession.append(game['Ball Possession'][team_ind])
            team_pass_accuracy.append(game['Passes %'][team_ind])
            team_goals.append(game['Goals'][team_ind])     
            opponent_total_shots.append(game['Total Shots'][opponent_ind])
            opponent_shots_inside_box.append(game['Shots insidebox'][opponent_ind])
            opponent_fouls.append(game['Fouls'][opponent_ind])
            opponent_corners.append(game['Corner Kicks'][opponent_ind])
            opponent_posession.append(game['Ball Possession'][opponent_ind])
            opponent_pass_accuracy.append(game['Passes %'][opponent_ind])
            opponent_goals.append(game['Goals'][opponent_ind])
            result_indicator_raw.append(game['Points'][team_ind])
             
        
        #sliding average of the raw feature lists above to create the final features
        team_total_shots_slide = running_mean(team_total_shots, games_slide)[:x]
        team_shots_inside_box_slide = running_mean(team_shots_inside_box, games_slide)[:x]
        team_fouls_slide = running_mean(team_fouls, games_slide)[:x]
        team_corners_slide = running_mean(team_corners, games_slide)[:x]
        team_posession_slide = running_mean(team_posession, games_slide)[:x]
        team_pass_accuracy_slide = running_mean(team_pass_accuracy, games_slide)[:x]
        team_goals_slide = running_mean(team_goals, games_slide)[:x]
        team_goals_target = team_goals[games_slide-xx:]
        opponent_total_shots_slide = running_mean(opponent_total_shots, games_slide)[:x]
        opponent_shots_inside_box_slide = running_mean( opponent_shots_inside_box, games_slide)[:x]
        opponent_fouls_slide = running_mean(opponent_fouls, games_slide)[:x]
        opponent_corners_slide = running_mean(opponent_corners, games_slide)[:x]
        opponent_posession_slide = running_mean(opponent_posession, games_slide)[:x]
        opponent_pass_accuracy_slide = running_mean(opponent_pass_accuracy, games_slide)[:x]
        opponent_goals_slide = running_mean(opponent_goals, games_slide)[:x]
        opponent_goals_target = opponent_goals[games_slide-xx:]
        fix_id_slide = team_fixture_id_dict[team_id][games_slide-xx:]
        result_indicator_slide = result_indicator_raw[games_slide-xx:]

    
        #appending over the iterables, the above variables will be overwritten with each iteration
        t_total_shots.extend(team_total_shots_slide)
        t_shots_inside_box.extend(team_shots_inside_box_slide)
        t_fouls.extend(team_fouls_slide)
        t_corners.extend(team_corners_slide)
        t_posession.extend(team_posession_slide)
        t_pass_accuracy.extend(team_pass_accuracy_slide)
        t_goals.extend(team_goals_slide)
        t_goals_target.extend(team_goals_target)
        o_total_shots.extend(opponent_total_shots_slide)
        o_shots_inside_box.extend(opponent_shots_inside_box_slide)
        o_fouls.extend(opponent_fouls_slide)
        o_corners.extend(opponent_corners_slide)
        o_posession.extend(opponent_posession_slide)
        o_pass_accuracy.extend(opponent_pass_accuracy_slide)
        o_goals.extend(opponent_goals_slide)
        o_goals_target.extend(opponent_goals_target)
        fix_id.extend(fix_id_slide)
        result_indicator.extend(result_indicator_slide)
        o_team_ID.append(team_id)

    
    #piecing together the results into a dataframe   
    df_ready_for_ml = pd.DataFrame({})  
    df_ready_for_ml['Team Av Shots'] = t_total_shots
    df_ready_for_ml['Team Av Shots Inside Box'] = t_shots_inside_box
    df_ready_for_ml['Team Av Fouls'] = t_fouls
    df_ready_for_ml['Team Av Corners'] = t_corners
    df_ready_for_ml['Team Av Possession'] = t_posession
    df_ready_for_ml['Team Av Pass Accuracy'] = t_pass_accuracy
    df_ready_for_ml['Team Av Goals'] = t_goals
    df_ready_for_ml['Opponent Av Shots'] = o_total_shots
    df_ready_for_ml['Opponent Av Shots Inside Box'] = o_shots_inside_box
    df_ready_for_ml['Opponent Av Fouls'] = o_fouls
    df_ready_for_ml['Opponent Av Corners'] = o_corners
    df_ready_for_ml['Opponent Av Possession'] = o_posession
    df_ready_for_ml['Opponent Av Goals'] = o_goals
    df_ready_for_ml['Opponent Av Pass Accuracy'] = o_pass_accuracy
    if not making_predictions:
        df_ready_for_ml['Team Goal Target'] = t_goals_target
        df_ready_for_ml['Opponent Goal Target'] = o_goals_target
        df_ready_for_ml['Target Fixture ID'] = fix_id
        df_ready_for_ml['Result Indicator'] = result_indicator
    if making_predictions:
        df_ready_for_ml['team_id'] = o_team_ID 
    
    #returning the dataframe
    return df_ready_for_ml






