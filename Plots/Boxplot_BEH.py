# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 20:39:24 2022

@author: James Barry

Purpose: Run Anovas and post hoc tests while creating a graph with the results

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
os.chdir("C:\\Users\jericho\OneDrive\Work Stuff\TOT_BEH_Results")


#%% Open the master file

master = pd.read_csv("Results_combined_percentages.csv", delimiter=",")

#%% Remove all columns except the ones specified in the double brackets
counts = master[['Subject', 'Stimuli Type', 'Freq_1_50_TOT', 'Freq_51_100_TOT', 'Freq_101_150_TOT', 'Freq_151_200_TOT', 'Freq_200+_times_TOT']]

#%%
# Define the threshold for data cleaning
threshold = 2.5

# Clean data in each column that is more than threshold times larger or smaller than the mean
for col in counts[['Freq_1_50_TOT', 'Freq_51_100_TOT', 'Freq_101_150_TOT', 'Freq_151_200_TOT', 'Freq_200+_times_TOT']]:
    mean = counts[col].mean()
    std = counts[col].std()
    upper_limit = mean + threshold * std
    lower_limit = mean - threshold * std
    counts.loc[counts[col] > upper_limit, col] = np.nan
    counts.loc[counts[col] < lower_limit, col] = np.nan

#%% Cleaning the csv ready for ANOVAS and boxplots

#RT_Times = RT_Times.rename(columns={'TOT': 'TOT', 'unTOT': 'UnTOT'})

#RT_Times = RT_Times.mask(RT_Times.sub(RT_Times.mean()).div(RT_Times.std()).abs().gt(2.5))

#%%
counts = pd.melt(counts, id_vars=['Subject','Stimuli Type'], value_vars=['Freq_1_50_TOT', 'Freq_51_100_TOT', 'Freq_101_150_TOT', 'Freq_151_200_TOT', 'Freq_200+_times_TOT'])

counts = counts.rename(columns={'variable': 'Frequency', 'value': 'Percent Response'})

#counts['Stimuli Type'] = counts['Stimuli Type'].replace(['face'], ['Face']).replace(['place'], ['Place'])


#%% create variables for the boxplots

data = counts
x = 'Frequency'
y = 'Percent Response'
order = ['Freq_1_50_TOT', 'Freq_51_100_TOT', 'Freq_101_150_TOT', 'Freq_151_200_TOT', 'Freq_200+_times_TOT']
hue = "Stimuli Type"
hue_order = ["Face", "Place"]
palette =  ["indigo", "darkgreen"]


#%% Set color for charts
PROPS = {
         'boxprops'       :{'edgecolor':'white'},
         'medianprops'    :{'color':'white'},
         'whiskerprops'   :{'color':'white'},
         'capprops'       :{'color':'white'}            
             
         }

#%% Create the boxplot with the significance lines episodic TOT

pairs = [
    
    #['4','3'],
    #['4','2'],
    #['4','1'],
    #['3','2'],
    #['3','1'],
    #['2','1'],
    
]

#%%
with sns.plotting_context("notebook", font_scale = 0.9):
        
    #Plot with seaborn
    sns.set(rc={"figure.dpi":500})
    plt.style.use('dark_background')
    count_raw = sns.boxplot(data = data,
                            x = x,
                            y = y,
                            hue = hue,
                            order = order,
                            palette = palette,
                            linewidth = 0.75, 
                            showfliers = False,
                            **PROPS)
    sns.despine(top=True, right=True)
    plt.figure(1, figsize=(80, 140), frameon=False, dpi=1000)
    plt.title("Frequency of Retrival - TOT", loc='center')
    plt.xticks([0, 1, 2, 3, 4], ['1-50 \n times', '51-100 \n times', '101-150 \n times', '151-200 \n times', '200+ \n times'])
    plt.axis([-1, 6, -10, 100])
    plt.grid(False)
    plt.xlabel("Frequency of Retrieval")
    plt.ylabel("Percent Response")
    
    
    #Add annotations
    #annotator = Annotator(PLACE_FINAL, pairs=pairs, **hue_plot_params)
    #annotator.configure(test="t-test_ind", comparisons_correction = "bonf", loc = "inside", text_format = "star", verbose = 2, line_offset = 5,
                        #line_height = 0.1, text_offset = 1, use_fixed_offset = True)
    
    #annotator.configure(test=None, loc = "inside", text_format = "star", verbose = 5,
                         #line_height = 0.05, text_offset = 0.1, line_offset = 0.05, color = 'white')
    #annotator.set_custom_annotations(["*"])
    #annotator.annotate()