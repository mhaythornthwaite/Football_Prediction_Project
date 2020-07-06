# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:11:31 2020

@author: mhayt
"""

#-------------------------------- API-FOOTBALL --------------------------------

#Access with: http://localhost:5000/

from flask import Flask, render_template
app = Flask(__name__)
import pickle


#------------------------------------ FLASK -----------------------------------


with open('../predictions/pl_predictions.csv', 'rb') as myFile:
    pl_pred = pickle.load(myFile)

g1_h = pl_pred['Home Team'][0]
g1_a = pl_pred['Away Team'][0]
g1_hw = pl_pred['Home Win'][0]
g1_d = pl_pred['Draw'][0]
g1_aw = pl_pred['Away Win'][0]
g1_h_l = pl_pred['Home Team Logo'][0]
g1_a_l = pl_pred['Away Team Logo'][0]


g2_h = pl_pred['Home Team'][1]
g2_a = pl_pred['Away Team'][1]
g2_hw = pl_pred['Home Win'][1]
g2_d = pl_pred['Draw'][1]
g2_aw = pl_pred['Away Win'][1]
g2_h_l = pl_pred['Home Team Logo'][1]
g2_a_l = pl_pred['Away Team Logo'][1]


g3_h = pl_pred['Home Team'][2]
g3_a = pl_pred['Away Team'][2]
g3_hw = pl_pred['Home Win'][2]
g3_d = pl_pred['Draw'][2]
g3_aw = pl_pred['Away Win'][2]
g3_h_l = pl_pred['Home Team Logo'][2]
g3_a_l = pl_pred['Away Team Logo'][2]


g4_h = pl_pred['Home Team'][3]
g4_a = pl_pred['Away Team'][3]
g4_hw = pl_pred['Home Win'][3]
g4_d = pl_pred['Draw'][3]
g4_aw = pl_pred['Away Win'][3]
g4_h_l = pl_pred['Home Team Logo'][3]
g4_a_l = pl_pred['Away Team Logo'][3]

g5_h = pl_pred['Home Team'][4]
g5_a = pl_pred['Away Team'][4]
g5_hw = pl_pred['Home Win'][4]
g5_d = pl_pred['Draw'][4]
g5_aw = pl_pred['Away Win'][4]
g5_h_l = pl_pred['Home Team Logo'][4]
g5_a_l = pl_pred['Away Team Logo'][4]


g6_h = pl_pred['Home Team'][5]
g6_a = pl_pred['Away Team'][5]
g6_hw = pl_pred['Home Win'][5]
g6_d = pl_pred['Draw'][5]
g6_aw = pl_pred['Away Win'][5]
g6_h_l = pl_pred['Home Team Logo'][5]
g6_a_l = pl_pred['Away Team Logo'][5]


g7_h = pl_pred['Home Team'][6]
g7_a = pl_pred['Away Team'][6]
g7_hw = pl_pred['Home Win'][6]
g7_d = pl_pred['Draw'][6]
g7_aw = pl_pred['Away Win'][6]
g7_h_l = pl_pred['Home Team Logo'][6]
g7_a_l = pl_pred['Away Team Logo'][6]

g8_h = pl_pred['Home Team'][7]
g8_a = pl_pred['Away Team'][7]
g8_hw = pl_pred['Home Win'][7]
g8_d = pl_pred['Draw'][7]
g8_aw = pl_pred['Away Win'][7]
g8_h_l = pl_pred['Home Team Logo'][7]
g8_a_l = pl_pred['Away Team Logo'][7]


g9_h = pl_pred['Home Team'][8]
g9_a = pl_pred['Away Team'][8]
g9_hw = pl_pred['Home Win'][8]
g9_d = pl_pred['Draw'][8]
g9_aw = pl_pred['Away Win'][8]
g9_h_l = pl_pred['Home Team Logo'][8]
g9_a_l = pl_pred['Away Team Logo'][8]


g10_h = pl_pred['Home Team'][9]
g10_a = pl_pred['Away Team'][9]
g10_hw = pl_pred['Home Win'][9]
g10_d = pl_pred['Draw'][9]
g10_aw = pl_pred['Away Win'][9]
g10_h_l = pl_pred['Home Team Logo'][9]
g10_a_l = pl_pred['Away Team Logo'][9]


@app.route('/')
def pass_game_1():
    return render_template('index.html', 
                           g1_h=g1_h, g1_a=g1_a, g1_hw=g1_hw, g1_d=g1_d, g1_aw=g1_aw, g1_h_l=g1_h_l, g1_a_l=g1_a_l,
                           g2_h=g2_h, g2_a=g2_a, g2_hw=g2_hw, g2_d=g2_d, g2_aw=g2_aw,
                           g3_h=g3_h, g3_a=g3_a, g3_hw=g3_hw, g3_d=g3_d, g3_aw=g3_aw,
                           g4_h=g4_h, g4_a=g4_a, g4_hw=g4_hw, g4_d=g4_d, g4_aw=g4_aw,
                           g5_h=g5_h, g5_a=g5_a, g5_hw=g5_hw, g5_d=g5_d, g5_aw=g5_aw,
                           g6_h=g6_h, g6_a=g6_a, g6_hw=g6_hw, g6_d=g6_d, g6_aw=g6_aw,
                           g7_h=g7_h, g7_a=g7_a, g7_hw=g7_hw, g7_d=g7_d, g7_aw=g7_aw,
                           g8_h=g8_h, g8_a=g8_a, g8_hw=g8_hw, g8_d=g8_d, g8_aw=g8_aw,
                           g9_h=g9_h, g9_a=g9_a, g9_hw=g9_hw, g9_d=g9_d, g9_aw=g9_aw,
                           g10_h=g10_h, g10_a=g10_a, g10_hw=g10_hw, g10_d=g10_d, g10_aw=g10_aw)




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

