# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:21:32 2020

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

with open('2019_prem_generated_clean/2019_prem_df_for_ml_5_v2.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('2019_prem_generated_clean/2019_prem_df_for_ml_10_v2.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

#standard variables for every figure
colourbar = 'winter'



#----------------------------------- FIGURE 1 ---------------------------------

#figure 1 - setting up the wrapper
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

fig.suptitle('Data Averaged Over 10 Games', y=0.99, fontsize=16, fontweight='bold');

#plotting the 6 figures
scat1 = ax1.scatter(df_ml_10['Team Av Shots Diff'], 
                   df_ml_10['Opponent Av Shots Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar);

scat2 = ax2.scatter(df_ml_10['Team Av Shots Inside Box Diff'], 
                   df_ml_10['Opponent Av Shots Inside Box Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar);

scat3 = ax3.scatter(df_ml_10['Team Av Fouls Diff'], 
                   df_ml_10['Opponent Av Fouls Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar);

scat4 = ax4.scatter(df_ml_10['Team Av Corners Diff'], 
                   df_ml_10['Opponent Av Corners Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar);


scat5 = ax5.scatter(df_ml_10['Team Av Pass Accuracy Diff'], 
                   df_ml_10['Opponent Av Pass Accuracy Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar);

scat6 = ax6.scatter(df_ml_10['Team Av Goal Diff'], 
                   df_ml_10['Opponent Av Goal Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar);

#setting axis and legend for all 6 figures
fig.tight_layout(pad=6)

ax1.set(xlabel='Team Average Shots Difference',
        ylabel='Opponent Average Shots');

ax2.set(xlabel='Team Average Shots Inside Box Difference',
        ylabel='Opponent Average Shots Inside Box Difference');

ax3.set(xlabel='Team Average Fouls Difference',
        ylabel='Opponent Average Fouls Difference');

ax4.set(xlabel='Team Average Corners Difference',
        ylabel='Opponent Average Corners Difference');

ax5.set(xlabel='Team Average Pass Accuracy % Difference',
        ylabel='Opponent Average Pass Accuracy % Difference');

ax6.set(xlabel='Team Average Goals Difference',
        ylabel='Opponent Average Goals Difference');

ax_iter = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in ax_iter:
    ax.legend(*scat2.legend_elements(), title='Target \n Team \n Result', loc='upper right', fontsize='small');
    ax.set_axisbelow(True)
    ax.grid(color='xkcd:light grey')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    #ax.plot(lims, lims, '--', color = '#FFAAAA')
    a_min = lims[0]
    a_max = lims[1]
    mult = lims[1] - lims[0]
    ax.plot([a_min, a_max], [a_min, a_max], '--', color = '#DD7E7E')
    #ax.plot([a_min, a_max], [a_min+0.3*mult, a_max+0.3*mult], '--', color = '#FFCECE')
    #ax.plot([a_min, a_max], [a_min-0.3*mult, a_max-0.3*mult], '--', color = '#FFCECE')


#saving figure
fig.savefig('figures/average_10_games_team_target_result.png')




#----------------------------------- FIGURE 2 ---------------------------------

#figure 1 - setting up the wrapper
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

fig.suptitle('Data Averaged Over 5 Games', y=0.99, fontsize=16, fontweight='bold');

#plotting the 6 figures
scat1 = ax1.scatter(df_ml_5['Team Av Shots Diff'], 
                   df_ml_5['Opponent Av Shots Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar);

scat2 = ax2.scatter(df_ml_5['Team Av Shots Inside Box Diff'], 
                   df_ml_5['Opponent Av Shots Inside Box Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar);

scat3 = ax3.scatter(df_ml_5['Team Av Fouls Diff'], 
                   df_ml_5['Opponent Av Fouls Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar);

scat4 = ax4.scatter(df_ml_5['Team Av Corners Diff'], 
                   df_ml_5['Opponent Av Corners Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar);


scat5 = ax5.scatter(df_ml_5['Team Av Pass Accuracy Diff'], 
                   df_ml_5['Opponent Av Pass Accuracy Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar);

scat6 = ax6.scatter(df_ml_5['Team Av Goal Diff'], 
                   df_ml_5['Opponent Av Goal Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar);

#setting axis and legend for all 6 figures
fig.tight_layout(pad=6)

ax1.set(xlabel='Team Average Shots Difference',
        ylabel='Opponent Average Shots');

ax2.set(xlabel='Team Average Shots Inside Box Difference',
        ylabel='Opponent Average Shots Inside Box Difference');

ax3.set(xlabel='Team Average Fouls Difference',
        ylabel='Opponent Average Fouls Difference');

ax4.set(xlabel='Team Average Corners Difference',
        ylabel='Opponent Average Corners Difference');

ax5.set(xlabel='Team Average Pass Accuracy % Difference',
        ylabel='Opponent Average Pass Accuracy % Difference');

ax6.set(xlabel='Team Average Goals Difference',
        ylabel='Opponent Average Goals Difference');

ax_iter = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in ax_iter:
    ax.legend(*scat2.legend_elements(), title='Target \n Team \n Result', loc='upper right', fontsize='small');
    ax.set_axisbelow(True)
    ax.grid(color='xkcd:light grey')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    #ax.plot(lims, lims, '--', color = '#FFAAAA')
    a_min = lims[0]
    a_max = lims[1]
    mult = lims[1] - lims[0]
    ax.plot([a_min, a_max], [a_min, a_max], '--', color = '#DD7E7E')
    #ax.plot([a_min, a_max], [a_min+0.3*mult, a_max+0.3*mult], '--', color = '#FFCECE')
    #ax.plot([a_min, a_max], [a_min-0.3*mult, a_max-0.3*mult], '--', color = '#FFCECE')


#saving figure
fig.savefig('figures/average_5_games_team_target_result.png')








