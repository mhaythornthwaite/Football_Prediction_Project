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


with open('../2019_prem_generated_clean/2019_prem_df_for_ml_10_v2.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

x = df_ml_10['Team Av Corners Diff'][45]

with open('../predictions/pl_predictions.csv', 'rb') as myFile:
    pl_pred = pickle.load(myFile)



@app.route('/')
def hello_world():
    x = round(df_ml_10['Team Av Corners Diff'][45], 2) + 48
    g1_h = pl_pred['Home Team'][0]
    g1_a = pl_pred['Away Team'][0]
    g1_hw = pl_pred['Home Win'][0]
    g1_d = pl_pred['Draw'][0]
    g1_aw = pl_pred['Away Win'][0]
    return render_template('index.html', x=x, g1_h=g1_h, g1_a=g1_a, g1_hw=g1_hw, g1_d=g1_d, g1_aw=g1_aw)

# =============================================================================
# @app.route('/')
# def hello_world():
#     return 'Hello, World! I\'m called Matt'
# =============================================================================

if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0', port = 5000)
    

#----- FAVICONS -----

'''
In our HTML we used the following to link a favicon icon. 

{{ url_for('static', filename='favicon_1.ico') }}

Note that we used the two {{ }}. This is the utilising the Jinja API within flask that tells the HTML to execte what is insde the brackets like a python command. Therefore typing {{ 4 + 5 }} will print 9 to the screen, something that standard HTML would not do
'''

