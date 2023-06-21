#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:14:07 2022

@author: jericho
"""
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib as mpl

# %% set directory
os.chdir(r'C:\\Users\\jericho\\OneDrive\\Work Stuff\\TOT_BEH_Results')

# %% import the master csv file
combined_results_master = pd.read_csv(r'RESULTS_COMBINED_MASTER.csv')

# %% Remove all columns except the ones specified in the double brackets
Response_Type = combined_results_master[['Subject', 'Stimuli_Type', 'Code']]


# %% Make a nice bar chart with percentages
sns.set(style="whitegrid", font_scale=1.2)

Response_Type_barplot = sns.barplot(x='Code',
                                    y=np.ones(len(Response_Type)),
                                    data=Response_Type,
                                    palette=('Blues'),
                                    order=["known", "TOT", "familiar", "unknown"],
                                    hue="Stimuli_Type",
                                    estimator=lambda x: len(x) * 100.0 / len(Response_Type),
                                    errorbar=("ci", 95),
                                    edgecolor="black")

Response_Type_barplot.spines['top'].set_visible(False)
Response_Type_barplot.spines['right'].set_visible(False)
Response_Type_barplot.spines['bottom'].set_visible(True)
Response_Type_barplot.spines['left'].set_visible(True)
Response_Type_barplot.set_xlabel("Response Type", fontsize=12)
Response_Type_barplot.set_ylabel("Response Frequency (%)", fontsize=12)
Response_Type_barplot.set_title("Overall Response Rate", loc='center', fontsize=16)
bars = Response_Type_barplot.containers[0]
bars2 = Response_Type_barplot.containers[1]
Response_Type_barplot.set_xticklabels(("Known", "TOT", "Familiar", "Unknown"))
Response_Type_barplot.tick_params(axis='x', labelsize=12)
Response_Type_barplot.tick_params(axis='y', labelsize=12)
plt.ylim(0, 40)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

for container in Response_Type_barplot.containers:
    for bar in container:
        height = bar.get_height()
        bar.set_height(height * 2)
        Response_Type_barplot.text(bar.get_x() + bar.get_width() / 2, height * 2,
                                   f'{height * 2:.2f}', ha='center', va='bottom', fontsize=12)


# %% Remove rows that we don't want
Code = Code.drop(Code[Code['Code'].str.match('TOT',na=False)].index)
Code = Code.drop(Code[Code['Code'].str.match('familiar',na=False)].index)
Code = Code.drop(Code[Code['Code'].str.match('unknown',na=False)].index)

# %% for some reason it helps to reset the index
Code = Code.reset_index(drop=True)

# %% Convert to string
Code.Frequency = Code.First_encountered.astype(str)

# %% Rename responses to match what they are
Code['First_encountered'] = Code['First_encountered'].replace({'1.0':'< 6'})
Code['First_encountered'] = Code['First_encountered'].replace({'2.0':'6-11'})
Code['First_encountered'] = Code['First_encountered'].replace({'3.0':'12-15'})
Code['First_encountered'] = Code['First_encountered'].replace({'4.0':'16-20'})
Code['First_encountered'] = Code['First_encountered'].replace({'5.0':'21+'})

# %% Make a nice bar chart with percentages
Codechart = sns.barplot(x='First_encountered',
                             y=np.ones(len(Code)), 
                             data=Code, 
                             palette=['indigo', 'darkgreen'],
                             hue = "Stimuli_Type",
                             estimator=lambda x: len(x) * 100.0 / len(Code),
                             errorbar=("ci",95),
                             edgecolor = "black")

Codechart.grid(False)
Codechart.spines['top'].set_visible(False)
Codechart.spines['right'].set_visible(False)
Codechart.spines['bottom'].set_visible(True)
Codechart.spines['left'].set_visible(True)
Codechart.set_xticklabels( ('< 6', '6-11', '12-15', '16-20', '21+') )
Codechart.set( xlabel = "Age of Aquisition", ylabel = "Response Frequency (percentages)")
plt.title("Age of Aquisition Retreival Rate- Known", loc='center')
bars = Codechart.containers[0]
bars2 = Codechart.containers[1]
Codechart.bar_label(Codechart.containers[0], fontsize=10, fmt="%.2f")
Codechart.bar_label(Codechart.containers[1], fontsize=10, fmt="%.2f")
plt.style.use('dark_background')
plt.axis([-1, 5, 0, 20])
plt.rcParams.update({'font.size': 14})
plt.figure(1, figsize=(80,140), frameon=False, dpi=1000)