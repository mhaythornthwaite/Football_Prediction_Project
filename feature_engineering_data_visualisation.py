# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:11:16 2020

@author: mhayt
"""


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#-------------------------- PRE-ML DATA VISUALISATION -------------------------

with open('2019_prem_generated_clean/2019_prem_df_for_ml_5.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('2019_prem_generated_clean/2019_prem_df_for_ml_10.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

#standard variables for every figure
colourbar = 'inferno'
target_goals_min = 0
target_goals_max = 4

#----------------------------------- FIGURE 1 ---------------------------------

#figure 1 - setting up the wrapper
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

fig.suptitle('Data Averaged Over 10 Games - Target = Team Goals', y=0.99, fontsize=16, fontweight='bold');

#plotting the 6 figures
scat1 = ax1.scatter(df_ml_10['Team Av Shots'], 
                   df_ml_10['Opponent Av Shots'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat2 = ax2.scatter(df_ml_10['Team Av Shots Inside Box'], 
                   df_ml_10['Opponent Av Shots Inside Box'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat3 = ax3.scatter(df_ml_10['Team Av Fouls'], 
                   df_ml_10['Opponent Av Fouls'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat4 = ax4.scatter(df_ml_10['Team Av Corners'], 
                   df_ml_10['Opponent Av Corners'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);


scat5 = ax5.scatter(df_ml_10['Team Av Pass Accuracy'], 
                   df_ml_10['Opponent Av Pass Accuracy'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat6 = ax6.scatter(df_ml_10['Team Av Goals'], 
                   df_ml_10['Opponent Av Goals'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

#setting axis and legend for all 6 figures
fig.tight_layout(pad=6)

ax1.set(xlabel='Team Average Shots',
        ylabel='Opponent Average Shots');

ax2.set(xlabel='Team Average Shots Inside Box',
        ylabel='Opponent Average Shots Inside Box');

ax3.set(xlabel='Team Average Fouls',
        ylabel='Opponent Average Fouls');

ax4.set(xlabel='Team Average Corners',
        ylabel='Opponent Average Corners');

ax5.set(xlabel='Team Average Pass Accuracy %',
        ylabel='Opponent Average Pass Accuracy %');

ax6.set(xlabel='Team Average Goals',
        ylabel='Opponent Average Goals');

ax_iter = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in ax_iter:
    ax.legend(*scat2.legend_elements(), title='Target \n Team \n Goals', loc='upper right', fontsize='small');
    ax.set_axisbelow(True)
    ax.grid(color='xkcd:light grey')

#saving figure
fig.savefig('figures/average_10_games_team_target_goals.png')


#----------------------------------- FIGURE 2 ---------------------------------

#figure 2 - setting up the wrapper
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

fig.suptitle('Data Averaged Over 10 Games - Target = Opponent Goals', y=0.99, fontsize=16, fontweight='bold');

#plotting the 6 figures
scat1 = ax1.scatter(df_ml_10['Team Av Shots'], 
                   df_ml_10['Opponent Av Shots'], 
                   c=df_ml_10['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat2 = ax2.scatter(df_ml_10['Team Av Shots Inside Box'], 
                   df_ml_10['Opponent Av Shots Inside Box'], 
                   c=df_ml_10['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat3 = ax3.scatter(df_ml_10['Team Av Fouls'], 
                   df_ml_10['Opponent Av Fouls'], 
                   c=df_ml_10['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat4 = ax4.scatter(df_ml_10['Team Av Corners'], 
                   df_ml_10['Opponent Av Corners'], 
                   c=df_ml_10['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);


scat5 = ax5.scatter(df_ml_10['Team Av Pass Accuracy'], 
                   df_ml_10['Opponent Av Pass Accuracy'], 
                   c=df_ml_10['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat6 = ax6.scatter(df_ml_10['Team Av Goals'], 
                   df_ml_10['Opponent Av Goals'], 
                   c=df_ml_10['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

#setting axis and legend for all 6 figures
fig.tight_layout(pad=6)

ax1.set(xlabel='Team Average Shots',
        ylabel='Opponent Average Shots');

ax2.set(xlabel='Team Average Shots Inside Box',
        ylabel='Opponent Average Shots Inside Box');

ax3.set(xlabel='Team Average Fouls',
        ylabel='Opponent Average Fouls');

ax4.set(xlabel='Team Average Corners',
        ylabel='Opponent Average Corners');

ax5.set(xlabel='Team Average Pass Accuracy %',
        ylabel='Opponent Average Pass Accuracy %');

ax6.set(xlabel='Team Average Goals',
        ylabel='Opponent Average Goals');

ax_iter = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in ax_iter:
    ax.legend(*scat2.legend_elements(), title='   Target \n Opponent \n    Goals', loc='upper right', fontsize='small');
    ax.set_axisbelow(True)
    ax.grid(color='xkcd:light grey')

#saving figure
fig.savefig('figures/average_10_games_opponent_target_goals.png')


#----------------------------------- FIGURE 3 ---------------------------------


#figure 3 - setting up the wrapper
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

fig.suptitle('Data Averaged Over 5 Games - Target = Team Goals', y=0.99, fontsize=16, fontweight='bold');

#plotting the 6 figures
scat1 = ax1.scatter(df_ml_5['Team Av Shots'], 
                   df_ml_5['Opponent Av Shots'], 
                   c=df_ml_5['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat2 = ax2.scatter(df_ml_5['Team Av Shots Inside Box'], 
                   df_ml_5['Opponent Av Shots Inside Box'], 
                   c=df_ml_5['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat3 = ax3.scatter(df_ml_5['Team Av Fouls'], 
                   df_ml_5['Opponent Av Fouls'], 
                   c=df_ml_5['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat4 = ax4.scatter(df_ml_5['Team Av Corners'], 
                   df_ml_5['Opponent Av Corners'], 
                   c=df_ml_5['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);


scat5 = ax5.scatter(df_ml_5['Team Av Pass Accuracy'], 
                   df_ml_5['Opponent Av Pass Accuracy'], 
                   c=df_ml_5['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat6 = ax6.scatter(df_ml_5['Team Av Goals'], 
                   df_ml_5['Opponent Av Goals'], 
                   c=df_ml_5['Team Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

#setting axis and legend for all 6 figures
fig.tight_layout(pad=6)

ax1.set(xlabel='Team Average Shots',
        ylabel='Opponent Average Shots');

ax2.set(xlabel='Team Average Shots Inside Box',
        ylabel='Opponent Average Shots Inside Box');

ax3.set(xlabel='Team Average Fouls',
        ylabel='Opponent Average Fouls');

ax4.set(xlabel='Team Average Corners',
        ylabel='Opponent Average Corners');

ax5.set(xlabel='Team Average Pass Accuracy %',
        ylabel='Opponent Average Pass Accuracy %');

ax6.set(xlabel='Team Average Goals',
        ylabel='Opponent Average Goals');

ax_iter = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in ax_iter:
    ax.legend(*scat2.legend_elements(), title='Target \n Team \n Goals', loc='upper right', fontsize='small');
    ax.set_axisbelow(True)
    ax.grid(color='xkcd:light grey')

#saving figure
fig.savefig('figures/average_5_games_team_target_goals.png')



#----------------------------------- FIGURE 4 ---------------------------------

#figure 4 - setting up the wrapper
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

fig.suptitle('Data Averaged Over 5 Games - Target = Opponent Goals', y=0.99, fontsize=16, fontweight='bold');

#plotting the 6 figures
scat1 = ax1.scatter(df_ml_5['Team Av Shots'], 
                   df_ml_5['Opponent Av Shots'], 
                   c=df_ml_5['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat2 = ax2.scatter(df_ml_5['Team Av Shots Inside Box'], 
                   df_ml_5['Opponent Av Shots Inside Box'], 
                   c=df_ml_5['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat3 = ax3.scatter(df_ml_5['Team Av Fouls'], 
                   df_ml_5['Opponent Av Fouls'], 
                   c=df_ml_5['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat4 = ax4.scatter(df_ml_5['Team Av Corners'], 
                   df_ml_5['Opponent Av Corners'], 
                   c=df_ml_5['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);


scat5 = ax5.scatter(df_ml_5['Team Av Pass Accuracy'], 
                   df_ml_5['Opponent Av Pass Accuracy'], 
                   c=df_ml_5['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

scat6 = ax6.scatter(df_ml_5['Team Av Goals'], 
                   df_ml_5['Opponent Av Goals'], 
                   c=df_ml_5['Opponent Goal Target'],
                   cmap = colourbar,
                   vmin = target_goals_min,
                   vmax = target_goals_max);

#setting axis and legend for all 6 figures
fig.tight_layout(pad=6)

ax1.set(xlabel='Team Average Shots',
        ylabel='Opponent Average Shots');

ax2.set(xlabel='Team Average Shots Inside Box',
        ylabel='Opponent Average Shots Inside Box');

ax3.set(xlabel='Team Average Fouls',
        ylabel='Opponent Average Fouls');

ax4.set(xlabel='Team Average Corners',
        ylabel='Opponent Average Corners');

ax5.set(xlabel='Team Average Pass Accuracy %',
        ylabel='Opponent Average Pass Accuracy %');

ax6.set(xlabel='Team Average Goals',
        ylabel='Opponent Average Goals');

ax_iter = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in ax_iter:
    ax.legend(*scat2.legend_elements(), title='   Target \n Opponent \n    Goals', loc='upper right', fontsize='small');
    ax.set_axisbelow(True)
    ax.grid(color='xkcd:light grey')

#saving figure
fig.savefig('figures/average_5_games_opponent_target_goals.png')




# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')