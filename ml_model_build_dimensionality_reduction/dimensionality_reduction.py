# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:25:15 2020

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

#!/usr/bin/python
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep)

import time
start=time.time()

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
from ml_functions.data_processing import scale_df, scree_plot
import matplotlib.pyplot as plt

plt.close('all')


#------------------------------- INPUT VARIABLES ------------------------------

df_10_saved_name = '2019_prem_df_for_ml_10_v2.txt'

plot_scree_plot = True
save_scree_plot = False

xplot_pc1_pc2 = True
save_pc1_pc2 = False

xplot_ld1_ld2 = True
save_ld1_ld2 = False


#------------------------ PRINCIPLE COMPONENT ANALYSIS ------------------------

with open(f'../prem_clean_fixtures_and_dataframes/{df_10_saved_name}', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)

#scaling dataframe to make all features to have zero mean and unit vector. This is essential prior to PCA as Euclidean distance is used.
df_ml_10 = scale_df(df_ml_10, list(range(14)), [14,15,16])


#creating target and feature df
x_10 = df_ml_10.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_10 = df_ml_10['Team Result Indicator']

# Printing contribution/variation from each priciple component.
n_components = 6
pca = PCA(n_components=n_components)
pca.fit(x_10)
pca_percentages = list(pca.explained_variance_ratio_)
pca_percentages = [element * 100 for element in pca_percentages]
for i in range(0, n_components, 1):
    print(f'PCA{i+1}:', round(pca_percentages[i], 2), '%')

if plot_scree_plot:
    fig = scree_plot(pca_percentages)
    if save_scree_plot:
        fig.savefig('figures/PCA_Scree_Plot_ml10.png')


# ---------- X-plot PC1 and PC2 ----------

#creating variables
pca_values = pca.fit_transform(x_10)

if xplot_pc1_pc2:
    
    #instantiating figure and plotting scatter
    fig, ax = plt.subplots()
    scat = ax.scatter(pca_values[:,0], 
                      pca_values[:,1], 
                      c=df_ml_10['Team Result Indicator'], 
                      cmap='winter');
    
    #fig plotting details
    fig.suptitle('PCA X-Plot', y=0.96, fontsize=16, fontweight='bold');
    ax.set(xlabel='PC1', ylabel='PC2');
    ax.legend(*scat.legend_elements(), 
              title='Target \n Team \n Result', 
              loc='upper right', 
              fontsize='small')
    ax.grid(color='xkcd:light grey')
    ax.set_axisbelow(True)
    
    if save_pc1_pc2:
       fig.savefig('figures/PC1_PC2_xplot_ml10.png')


#------------------------ LINEAR DISCRIMINANT ANALYSIS ------------------------

#instantiating and fitting LDA classifier, for the purpose of dimensionality reduction.
clf = LinearDiscriminantAnalysis(n_components=2)
clf.fit(x_10, y_10)


# ---------- X-plot LDA1 and LDA2 ----------

#creating variables
lda_values = clf.fit_transform(x_10, y_10)

if xplot_ld1_ld2:
    
    #instantiating figure and plotting scatter
    fig, ax = plt.subplots()
    scat = ax.scatter(lda_values[:,0], 
                      lda_values[:,1], 
                      c=df_ml_10['Team Result Indicator'], 
                      cmap='winter');
    
    #fig plotting details
    fig.suptitle('LDA X-Plot', y=0.96, fontsize=16, fontweight='bold');
    ax.set(xlabel='LD1', ylabel='LD2');
    ax.legend(*scat.legend_elements(), 
              title='Target \n Team \n Result', 
              loc='upper right', 
              fontsize='small')
    
    ax.grid(color='xkcd:light grey')
    ax.set_axisbelow(True)
    
    if save_ld1_ld2:
        fig.savefig('figures/LDA_xplot_ml10.png')


#------------------------------------ END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
