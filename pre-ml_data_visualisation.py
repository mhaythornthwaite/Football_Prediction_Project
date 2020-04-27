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

normal_dist = np.random.normal(40, 10, 1000)
normal_dist = np.round(normal_dist, 2)

#subplots option 1 (recommended)
fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11, ax12, ax13, ax14)) = plt.subplots(nrows = 2, ncols = 7, figsize = (19, 6))
ax1.scatter(df_ml_10['Team Av Shots'], df_ml_10['Team Goal Target']);
ax2.scatter(df_ml_10['Team Av Shots Inside Box'], df_ml_10['Team Goal Target']);
ax3.scatter(df_ml_10['Team Av Fouls'], df_ml_10['Team Goal Target']);
ax4.scatter(df_ml_10['Team Av Corners'], df_ml_10['Team Goal Target']);
ax5.scatter(df_ml_10['Team Av Possession'], df_ml_10['Team Goal Target']);
ax6.scatter(df_ml_10['Team Av Pass Accuracy'], df_ml_10['Team Goal Target']);
ax7.scatter(df_ml_10['Team Av Goals'], df_ml_10['Team Goal Target']);
ax8.scatter(df_ml_10['Opponent Av Shots'], df_ml_10['Team Goal Target']);
ax9.scatter(df_ml_10['Opponent Av Shots Inside Box'], df_ml_10['Team Goal Target']);
ax10.scatter(df_ml_10['Opponent Av Fouls'], df_ml_10['Team Goal Target']);
ax11.scatter(df_ml_10['Opponent Av Corners'], df_ml_10['Team Goal Target']);
ax12.scatter(df_ml_10['Opponent Av Possession'], df_ml_10['Team Goal Target']);
ax13.scatter(df_ml_10['Opponent Av Goals'], df_ml_10['Team Goal Target']);
ax14.scatter(df_ml_10['Opponent Av Pass Accuracy'], df_ml_10['Team Goal Target']);

fig.tight_layout(pad=2.0, h_pad=2)



#figure 2
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

#plotting the 6 figures
scat1 = ax1.scatter(df_ml_10['Team Av Shots'], 
                   df_ml_10['Opponent Av Shots'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap='inferno',
                   vmin = 0,
                   vmax = 4);

scat2 = ax2.scatter(df_ml_10['Team Av Shots Inside Box'], 
                   df_ml_10['Opponent Av Shots Inside Box'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap='inferno',
                   vmin = 0,
                   vmax = 4);

scat3 = ax3.scatter(df_ml_10['Team Av Fouls'], 
                   df_ml_10['Opponent Av Fouls'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap='inferno',
                   vmin = 0,
                   vmax = 4);

scat4 = ax4.scatter(df_ml_10['Team Av Corners'], 
                   df_ml_10['Opponent Av Corners'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap='inferno',
                   vmin = 0,
                   vmax = 4);


scat5 = ax5.scatter(df_ml_10['Team Av Pass Accuracy'], 
                   df_ml_10['Opponent Av Pass Accuracy'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap='inferno',
                   vmin = 0,
                   vmax = 4);

scat6 = ax6.scatter(df_ml_10['Team Av Goals'], 
                   df_ml_10['Opponent Av Goals'], 
                   c=df_ml_10['Team Goal Target'],
                   cmap='inferno',
                   vmin = 0,
                   vmax = 4);

#setting axis and legend for all 6 figures




df_ml_10['Team Av Shots']
df_ml_10['Team Av Shots Inside Box']
df_ml_10['Team Av Fouls']
df_ml_10['Team Av Corners']
df_ml_10['Team Av Possession']
df_ml_10['Team Av Pass Accuracy']
df_ml_10['Team Av Goals']

df_ml_10['Opponent Av Shots']
df_ml_10['Opponent Av Shots Inside Box']
df_ml_10['Opponent Av Fouls']
df_ml_10['Opponent Av Corners']
df_ml_10['Opponent Av Possession']
df_ml_10['Opponent Av Goals']
df_ml_10['Opponent Av Pass Accuracy']

df_ml_10['Team Goal Target']
df_ml_10['Opponent Goal Target']





df_ml_5['Team Av Shots']
df_ml_5['Team Av Shots Inside Box']
df_ml_5['Team Av Fouls']
df_ml_5['Team Av Corners']
df_ml_5['Team Av Possession']
df_ml_5['Team Av Pass Accuracy']
df_ml_5['Team Av Goals']
df_ml_5['Opponent Av Shots']
df_ml_5['Opponent Av Shots Inside Box']
df_ml_5['Opponent Av Fouls']
df_ml_5['Opponent Av Corners']
df_ml_10['Opponent Av Possession']
df_ml_10['Opponent Av Goals']
df_ml_10['Opponent Av Pass Accuracy']
df_ml_10['Team Goal Target']
df_ml_10['Opponent Goal Target']


# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')