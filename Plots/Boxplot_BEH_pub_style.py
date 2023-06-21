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

# %% set directory
os.chdir("C:\\Users\\jericho\\OneDrive\\Work Stuff\\TOT_BEH_Results")


# %% Open the master file

master_box = pd.read_csv("Results_combined_percentages.csv", delimiter=",")

# %% Remove all columns except the ones specified in the double brackets
counts_response = master_box[['Subject', 'Stimuli Type', 'TOT', 'Familiar', 'Known', 'Unknown']]

# %%
# Define the threshold for data cleaning
threshold = 2.5

# Clean data in each column that is more than threshold times larger or smaller than the mean
for col in counts_response[['TOT', 'Familiar', 'Known', 'Unknown']]:
    mean = counts_response[col].mean()
    std = counts_response[col].std()
    upper_limit = mean + threshold * std
    lower_limit = mean - threshold * std
    counts_response.loc[counts_response[col] > upper_limit, col] = np.nan
    counts_response.loc[counts_response[col] < lower_limit, col] = np.nan


# %%
counts_response = pd.melt(counts_response, id_vars=['Subject', 'Stimuli Type'], value_vars=['TOT', 'Familiar', 'Known', 'Unknown'])

counts_response = counts_response.rename(columns={'variable': 'Frequency', 'value': 'Percent Response'})

# %% create variables for the boxplots

data = counts_response
x = 'Frequency'
y = 'Percent Response'
order = ['Known', 'TOT', 'Familiar', 'Unknown']
hue = "Stimuli Type"
hue_order = ["Face", "Place"]
palette = ("Blues")


# %% Set color for charts
PROPS = {
    'boxprops': {'edgecolor': 'black'},
    'medianprops': {'color': 'black'},
    'whiskerprops': {'color': 'black'},
    'capprops': {'color': 'black'}

}


# %%
with sns.plotting_context("notebook", font_scale=0.9):
    sns.set(rc={"figure.dpi": 500})
    sns.set(style="whitegrid", font_scale=1.2)
    count_raw = sns.boxplot(data=data,
                            x=x,
                            y=y,
                            hue=hue,
                            order=order,
                            palette=palette,
                            linewidth=0.75,
                            showfliers=False,
                            showmeans=True,
                            meanline=True,
                            meanprops={"color": "red"},
                            **PROPS)
    count_raw.spines['top'].set_visible(False)
    count_raw.spines['right'].set_visible(False)
    count_raw.spines['bottom'].set_visible(True)
    count_raw.spines['left'].set_visible(True)
    count_raw.set_xlabel("Response Type", fontsize=12)
    count_raw.set_ylabel("Response Frequency (%)", fontsize=12)
    count_raw.set_title("Overall Response Rate", loc='center', fontsize=16)
    count_raw.set_xticklabels(("Known", "TOT", "Familiar", "Unknown"))
    count_raw.tick_params(axis='x', labelsize=12)
    count_raw.tick_params(axis='y', labelsize=12)
    plt.ylim(0, 80)

    # Annotate mean value above the mean line
    for line in count_raw.lines[4::6]:
        x = line.get_xdata().mean()  # X-coordinate of the line's center
        y = line.get_ydata()[0]  # Y-coordinate of the line
        count_raw.text(x, y+-1, f"{y:.2f}", ha='center', va='top', fontsize=10)

    # Annotate mean value above the mean line
    for line in count_raw.lines[5::6]:
        x = line.get_xdata().mean()  # X-coordinate of the line's center
        y = line.get_ydata()[0]  # Y-coordinate of the line
        count_raw.text(x, y+0.5, f"{y:.2f}", ha='center', va='bottom', fontsize=10, color="Red")

    for line in count_raw.lines[2::6]:
        x = line.get_xdata().mean()  # X-coordinate of the line's center
        y = line.get_ydata()[0]  # Y-coordinate of the line (minimum)
        count_raw.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=10)

    for line in count_raw.lines[3::6]:
        x = line.get_xdata().mean()  # X-coordinate of the line's center
        y = line.get_ydata()[0]  # Y-coordinate of the line (maximum)
        count_raw.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=10)


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
