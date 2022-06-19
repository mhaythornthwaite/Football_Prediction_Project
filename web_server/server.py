# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:11:31 2020

@author: mhayt
"""

#-------------------------------- API-FOOTBALL --------------------------------

#Access with: http://localhost:5000/

from flask import Flask, render_template
import pickle
from datetime import datetime
app = Flask(__name__, static_url_path='/static')

#------------------------------------ FLASK -----------------------------------


with open('../predictions/pl_predictions.csv', 'rb') as myFile:
    pl_pred = pickle.load(myFile)
    
with open('../prem_clean_fixtures_and_dataframes/2019_2020_2021_2022_additional_stats_dict.txt', 'rb') as myFile:
    additional_stats_dict = pickle.load(myFile)    


#with open('/home/matthaythornthwaite/Football_Prediction_Project/web_server/pl_predictions.csv', 'rb') as myFile:
#    pl_pred = pickle.load(myFile)

#removing all past predictions if they still exist in the predictions df
current_date = datetime.today().strftime('%Y-%m-%d')
for j in range(len(pl_pred)):
    game_date = pl_pred['Game Date'].loc[j]
    if game_date < current_date:
        pl_pred = pl_pred.drop([j], axis=0)
pl_pred = pl_pred.reset_index(drop=True)        


#creating our iterator that we will use in the for loop in our index file.
max_display_games = 40
iterator_len = len(pl_pred) - 1
if iterator_len > max_display_games:
    iterator_len = max_display_games
iterator = range(iterator_len)

#creating our iterator that we will use in the for loop in our index file. Checking first that there is enough data.
max_additional_display_games = 5
dict_keys = list(additional_stats_dict.keys())
min_length = 100
for i in dict_keys:
    df_len = len(additional_stats_dict[i])
    if df_len < min_length:
        min_length = df_len
if max_additional_display_games > min_length:
    max_additional_display_games = min_length
iterator2 = range(max_additional_display_games)


@app.route('/')
def pass_game_1():
    return render_template('index.html',
                           pl_pred=pl_pred, 
                           iterator=iterator,
                           iterator2=iterator2,
                           additional_stats_dict=additional_stats_dict)



# =============================================================================
# @app.route('/')
# def hello_world():
#     return 'Hello, World! I\'m called Matt'
# =============================================================================

if __name__ == '__main__':
    #app.debug = True
    app.run(host = '0.0.0.0', port = 5000)
    

#----- FAVICONS -----

'''
In our HTML we used the following to link a favicon icon. 

{{ url_for('static', filename='favicon_1.ico') }}

Note that we used the two {{ }}. This is the utilising the Jinja API within flask that tells the HTML to execte what is insde the brackets like a python command. Therefore typing {{ 4 + 5 }} will print 9 to the screen, something that standard HTML would not do
'''

