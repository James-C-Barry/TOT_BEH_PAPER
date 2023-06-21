
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:31:10 2023

@author: jericho

logistic regressions - Publication Friendly

"""
# %% Imports

import pandas as pd
import seaborn as sns
import numpy as np
import os
import pingouin as pg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib import pyplot
import matplotlib.pyplot as plt
# import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# %% set working directory
os.chdir('C:\\Users\\jericho\\OneDrive\\Work Stuff\\TOT_BEH_Results')

# %% Open master file
master_faces = pd.read_csv(r'RESULTS_FACES_MASTER.csv')
master_places = pd.read_csv(r'RESULTS_PLACES_MASTER.csv')

# %% Clean the master file leaving the subject code and questions.
# Then removing the familiar and unknown responses and removing rows that contain no data

faces = master_faces[["Subject", "Code", "1=correct/2=incorrect", "First_encountered", "Last_Encountered", "Frequency"]]
faces = faces.drop(faces[faces['Code'].str.match('familiar', na=False)].index).drop(faces[faces['Code'].str.match('unknown', na=False)].index)
faces = faces.drop(faces[(faces['Code'] == "known") & (faces['1=correct/2=incorrect'] == "incorrect")].index)
faces = faces.dropna(axis=0, subset=["First_encountered", "Last_Encountered", "Frequency"]).reset_index(drop=True)

places = master_places[["Subject", "Code", "1=correct/2=incorrect", "First_encountered", "Last_Encountered", "Frequency"]]
places = places.drop(places[places['Code'].str.match('familiar', na=False)].index).drop(places[places['Code'].str.match('unknown', na=False)].index)
places = places.drop(places[(places['Code'] == "known") & (places['1=correct/2=incorrect'] == "incorrect")].index)
places = places.dropna(axis=0, subset=["First_encountered", "Last_Encountered", "Frequency"]).reset_index(drop=True)


# %% Check to see if there are any null values
face_heat = sns.heatmap(faces.isnull())
plt.style.use('dark_background')
face_heat.set_title("Faces")


# %%
place_heat = sns.heatmap(places.isnull())
plt.style.use('dark_background')
place_heat.set_title("Places")

# %% Convert all numbers to strings
faces['First_encountered'] = faces['First_encountered'].astype(str)
faces['Last_Encountered'] = faces['Last_Encountered'].astype(str)
faces['Frequency'] = faces['Frequency'].astype(str)

places['First_encountered'] = places['First_encountered'].astype(str)
places['Last_Encountered'] = places['Last_Encountered'].astype(str)
places['Frequency'] = places['Frequency'].astype(str)

# %% Rename all the responses in the column
faces['First_encountered'] = faces['First_encountered'].replace(['1.0',
                                                                 '2.0',
                                                                 '3.0',
                                                                 '4.0',
                                                                 '5.0'], ['<6',
                                                                          '6-11',
                                                                          '12-15',
                                                                          '16-20',
                                                                          '21+'])

faces['Last_Encountered'] = faces['Last_Encountered'].replace(['1.0',
                                                               '2.0',
                                                               '3.0',
                                                               '4.0',
                                                               '5.0'], ['1 week ago',
                                                                        '1 month ago',
                                                                        '6 months ago',
                                                                        '1 year ago',
                                                                        'more than 2 years'])

faces['Frequency'] = faces['Frequency'].replace(['1.0',
                                                 '2.0',
                                                 '3.0',
                                                 '4.0',
                                                 '5.0'], ['1-50 times',
                                                          '51-100 times',
                                                          '101-150 times',
                                                          '151-200 times',
                                                          '200+times'])

places['First_encountered'] = places['First_encountered'].replace(['1.0',
                                                                   '2.0',
                                                                   '3.0',
                                                                   '4.0',
                                                                   '5.0'], ['<6',
                                                                            '6-11',
                                                                            '12-15',
                                                                            '16-20',
                                                                            '21+'])

places['Last_Encountered'] = places['Last_Encountered'].replace(['1.0',
                                                                 '2.0',
                                                                 '3.0',
                                                                 '4.0',
                                                                 '5.0'], ['1 week ago',
                                                                          '1 month ago',
                                                                          '6 months ago',
                                                                          '1 year ago',
                                                                          'more than 2 years'])

places['Frequency'] = places['Frequency'].replace(['1.0',
                                                   '2.0',
                                                   '3.0',
                                                   '4.0',
                                                   '5.0'], ['1-50 times',
                                                            '51-100 times',
                                                            '101-150 times',
                                                            '151-200 times',
                                                            '200+times'])

# %% Output the varibale visualisation of overall responses - FACES

sns.set(style="whitegrid", font_scale=1.2)

# Create the bar plot
face_overall_bar = sns.barplot(
                               x='Code',
                               y=np.ones(len(faces)),
                               data=faces,
                               palette='Blues',
                               order=["known", "TOT"],
                               estimator=lambda x: len(x) * 100.0 / len(faces),
                               edgecolor="black"
                               )

# Set title and labels with larger font size
face_overall_bar.set_title("Percentage of TOT and Known Responses - Faces", fontsize=20)
face_overall_bar.set_xlabel("Code", fontsize=16)
face_overall_bar.set_ylabel("Percentage", fontsize=16)
plt.ylim(0, 100)

# Set tick label font size for x-axis and y-axis
face_overall_bar.tick_params(axis='x', labelsize=16)
face_overall_bar.tick_params(axis='y', labelsize=16)

# Plot bar labels with larger font size
for container in face_overall_bar.containers:
    for bar in container:
        height = bar.get_height()
        face_overall_bar.text(
                              bar.get_x() + bar.get_width() / 2,
                              height,
                              f'{height:.2f}%',  # Format the value as needed
                              ha='center',
                              va='bottom',
                              fontsize=18  # Adjust the font size as desired
                              )

# Remove the top and right spines
sns.despine()

# Adjust the figure size as needed
plt.figure(figsize=(8, 6))

# %% Output the varibale visualisation of overall responses - PlACES

sns.set(style="whitegrid", font_scale=1.2)

# Create the bar plot
place_overall_bar = sns.barplot(
                               x='Code',
                               y=np.ones(len(places)),
                               data=places,
                               palette='Blues',
                               order=["known", "TOT"],
                               estimator=lambda x: len(x) * 100.0 / len(places),
                               edgecolor="black"
                               )

# Set title and labels with larger font size
place_overall_bar.set_title("Percentage of TOT and Known Responses - places", fontsize=20)
place_overall_bar.set_xlabel("Code", fontsize=16)
place_overall_bar.set_ylabel("Percentage", fontsize=16)
plt.ylim(0, 100)

# Set tick label font size for x-axis and y-axis
place_overall_bar.tick_params(axis='x', labelsize=16)
place_overall_bar.tick_params(axis='y', labelsize=16)

# Plot bar labels with larger font size
for container in place_overall_bar.containers:
    for bar in container:
        height = bar.get_height()
        place_overall_bar.text(
                              bar.get_x() + bar.get_width() / 2,
                              height,
                              f'{height:.2f}%',  # Format the value as needed
                              ha='center',
                              va='bottom',
                              fontsize=18  # Adjust the font size as desired
                              )

# Remove the top and right spines
sns.despine()

# Adjust the figure size as needed
plt.figure(figsize=(8, 6))
