#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:54:04 2023

@author: jbarry
"""
import os
import pandas as pd
import seaborn as sns
import pingouin as pg
import numpy as np
import utils
from scipy import stats
import statannotations
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt

#%% set directory
os.chdir("C:\\Users\\jericho\\OneDrive\\Work Stuff\\TOT_BEH_Results")


#%% Open the master file

BEH_FACE_Results = pd.read_csv("Results_faces_raw_values.csv")
BEH_PLACE_Results = pd.read_csv("Results_places_raw_values.csv")

# Define the threshold for data cleaning
threshold = 2.5

# Clean data in each column that is more than threshold times larger or smaller than the mean
for col in BEH_FACE_Results[['TOT','Familiar','Known','Unknown']]:
    mean = BEH_FACE_Results[col].mean()
    std = BEH_FACE_Results[col].std()
    upper_limit = mean + threshold * std
    lower_limit = mean - threshold * std
    BEH_FACE_Results.loc[BEH_FACE_Results[col] > upper_limit, col] = np.nan
    BEH_FACE_Results.loc[BEH_FACE_Results[col] < lower_limit, col] = np.nan
    
# Clean data in each column that is more than threshold times larger or smaller than the mean
for col in BEH_PLACE_Results[['TOT','Familiar','Known','Unknown']]:
    mean = BEH_PLACE_Results[col].mean()
    std = BEH_PLACE_Results[col].std()
    upper_limit = mean + threshold * std
    lower_limit = mean - threshold * std
    BEH_PLACE_Results.loc[BEH_PLACE_Results[col] > upper_limit, col] = np.nan
    BEH_PLACE_Results.loc[BEH_PLACE_Results[col] < lower_limit, col] = np.nan

#%% Remove all columns except the ones specified in the double brackets
BEH_FACE_Results = BEH_FACE_Results[['Subject', 'Age', 'Stimuli Type', 'TOT', 'Familiar', 'Known', 'Unknown']]
BEH_PLACE_Results = BEH_PLACE_Results[['Subject', 'Age','Stimuli Type','TOT', 'Familiar', 'Known', 'Unknown']]
#%% Cleaning the csv ready for ANOVAS and boxplots

cols = ['TOT', 'Familiar', 'Known', 'Unknown']


#%%
FACE_FINAL = pd.melt(BEH_FACE_Results, id_vars=[('Age')], value_vars=['TOT', 'Familiar', 'Known', 'Unknown'])

FACE_FINAL = FACE_FINAL.rename(columns={'variable': 'response type', 'value': 'Raw No. of responses'})

#%%
PLACE_FINAL = pd.melt(BEH_PLACE_Results, id_vars=[('Age')], value_vars=['TOT', 'Familiar', 'Known', 'Unknown'])

PLACE_FINAL = PLACE_FINAL.rename(columns={'variable': 'response type', 'value': 'Raw No. of responses'})

#%%
with sns.plotting_context("notebook", font_scale = 0.9):
        
    #Plot with seaborn
    sns.set(rc={"figure.dpi":500})
    plt.style.use('dark_background')
    age_response_place = sns.lmplot(data = FACE_FINAL,
                            x = 'Age',
                            y = 'Raw No. of responses',
                            hue="response type")
    sns.despine(top=True, right=True)
    plt.figure(1, figsize=(80, 140), frameon=False, dpi=1000)
    plt.title("Response X Age Correlations (FACES)", loc='center')
    plt.axis([18, 35, 0, 200])
    plt.grid(False)
    plt.xlabel("Age")
    plt.ylabel("Number of Responses")