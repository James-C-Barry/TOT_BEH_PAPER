#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:31:10 2023

@author: jericho

Lets see if this works for logistic regressions

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

# %% Output the varibale visualisation of overall responses

plt.style.use('dark_background')
sns.despine(top=True, right=True)
plt.grid(False)

face_overall_bar = sns.barplot(x='Code',
                               y=np.ones(len(faces)),
                               data=faces,
                               palette='twilight',
                               order=["known", "TOT"],
                               estimator=lambda x: len(x) * 100.0 / len(faces),
                               edgecolor="black")

face_overall_bar.set_title("Percentage of TOT and Known Responses - Faces", fontsize=16)
face_overall_bar.set_xlabel("Code", fontsize=14)
face_overall_bar.set_ylabel("Percentage", fontsize=14)
face_overall_bar.tick_params(axis='x', labelsize=12)
face_overall_bar.tick_params(axis='y', labelsize=12)

# Plot bar labels
for container in face_overall_bar.containers:
    for bar in container:
        height = bar.get_height()
        face_overall_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}%',  # Format the value as needed
            ha='center',
            va='bottom',
            fontsize=14  # Adjust the font size as desired
        )

# %% Output the varibale visualisation of overall responses

plt.style.use('dark_background')
sns.despine(top=True, right=True)
plt.grid(False)

place_overall_bar = sns.barplot(x='Code',
                               y=np.ones(len(places)),
                               data=places,
                               palette='twilight',
                               order=["known", "TOT"],
                               estimator=lambda x: len(x) * 100.0 / len(places),
                               edgecolor="black")

place_overall_bar.set_title("Percentage of TOT and Known Responses - places", fontsize=16)
place_overall_bar.set_xlabel("Code", fontsize=14)
place_overall_bar.set_ylabel("Percentage", fontsize=14)
place_overall_bar.tick_params(axis='x', labelsize=12)
place_overall_bar.tick_params(axis='y', labelsize=12)

# Plot bar labels
for container in place_overall_bar.containers:
    for bar in container:
        height = bar.get_height()
        place_overall_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}%',  # Format the value as needed
            ha='center',
            va='bottom',
            fontsize=14  # Adjust the font size as desired
        )

# %% Output the varibale visualisation of overall responses split between first encountered
face_FE_Bar = sns.barplot(x='Code',
                          y=np.ones(len(faces)),
                          data=faces,
                          hue="First_encountered",
                          order=["known", "TOT"],
                          palette='twilight',
                          estimator=lambda x: len(x) * 100.0 / len(faces),
                          edgecolor="black").set(title="Counts of first encountered split by response - Faces")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# %% Output the varibale visualisation of overall responses split between first encountered
place_FE_bar = sns.barplot(x='Code',
                           y=np.ones(len(places)),
                           data=places,
                           hue="First_encountered",
                           order=["known", "TOT"],
                           palette='twilight',
                           estimator=lambda x: len(x) * 100.0 / len(places),
                           edgecolor="black").set(title="Counts of first encountered split by response - Places")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# %% Output the varibale visualisation of overall responses split between last encountered
face_LE_Bar = sns.barplot(x='Code',
                          y=np.ones(len(faces)),
                          data=faces,
                          hue="Last_Encountered",
                          order=["known", "TOT"],
                          palette='twilight',
                          estimator=lambda x: len(x) * 100.0 / len(faces),
                          edgecolor="black").set(title="Counts of last encountered split by response - Faces")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# %% Output the varibale visualisation of overall responses split between last encountered
place_LE_Bar = sns.barplot(x='Code',
                           y=np.ones(len(places)),
                           data=places,
                           hue="Last_Encountered",
                           palette='twilight',
                           estimator=lambda x: len(x) * 100.0 / len(places),
                           order=["known", "TOT"],
                           edgecolor="black").set(title="Counts of last encountered split by response - Places")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# %% Output the varibale visualisation of overall responses split between frequency
face_FREQ_Bar = sns.barplot(x='Code',
                            y=np.ones(len(faces)),
                            data=faces,
                            hue="Frequency",
                            palette='twilight',
                            estimator=lambda x: len(x) * 100.0 / len(faces),
                            order=["known", "TOT"],
                            edgecolor="black").set(title="Counts of Frequency split by response")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# %% Output the varibale visualisation of overall responses split between frequency
place_FREQ_Bar = sns.barplot(x='Code',
                             y=np.ones(len(places)),
                             data=places,
                             hue="Frequency",
                             palette='twilight',
                             estimator=lambda x: len(x) * 100.0 / len(places),
                             order=["known", "TOT"],
                             edgecolor="black").set(title="Counts of Frequency split by response")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# %%
# Separate majority and minority classes
faces_known = faces[faces.Code == "known"]
faces_TOT = faces[faces.Code == "TOT"]

# Upsample minority class
faces_TOT_upsampled = resample(faces_TOT,
                               replace=True,      # sample with replacement
                               n_samples=len(faces_known))   # to match majority class

# Downsample majority class
# faces_TOT_downsampled = resample(faces_known,
#                                 replace=True,      # sample with replacement
#                                 n_samples=len(faces_TOT))   # to match minority class


# Combine majority class with upsampled minority class
faces_upsampled = pd.concat([faces_known, faces_TOT_upsampled])

# Combine minority class with upsampled majority class
# faces_downsampled = pd.concat([faces_known, faces_TOT_downsampled])


# Separate majority and minority classes
places_known = places[places.Code == "known"]
places_TOT = places[places.Code == "TOT"]

# Upsample minority class
places_TOT_upsampled = resample(places_TOT,
                                replace=True,      # sample with replacement
                                n_samples=len(places_known))   # to match majority class

# Downsample minority class
# places_TOT_downsampled = resample(places_known,
#                                  replace=True,      # sample with replacement
#                                  n_samples=len(places_TOT))   # to match minority class


# Combine majority class with upsampled minority class
places_upsampled = pd.concat([places_known, places_TOT_upsampled])

# Combine majority class with upsampled minority class
# places_downsampled = pd.concat([places_known, places_TOT_downsampled])

# %% Now to get the dummy variables for everything - THIS IS FOR UPSAMPLING

First_encounted_faces = pd.get_dummies(faces_upsampled["First_encountered"], drop_first=False, prefix="FE")
Last_Encountered_faces = pd.get_dummies(faces_upsampled["Last_Encountered"], drop_first=False, prefix="LE")
Frequency_faces = pd.get_dummies(faces_upsampled["Frequency"], drop_first=False, prefix="FREQ")

First_encounted_places = pd.get_dummies(places_upsampled["First_encountered"], drop_first=False, prefix="FE")
Last_Encountered_places = pd.get_dummies(places_upsampled["Last_Encountered"], drop_first=False, prefix="LE")
Frequency_places = pd.get_dummies(places_upsampled["Frequency"], drop_first=False, prefix="FREQ")

# %% Drop columns that are not needed anymore
faces_upsampled.drop(["First_encountered", "Last_Encountered", "Frequency"], axis=1, inplace=True)

places_upsampled.drop(["First_encountered", "Last_Encountered", "Frequency"], axis=1, inplace=True)

# %% Concatenate the dataframe with the new columns that contain the dummy variables
faces_dummied = pd.concat([faces_upsampled, First_encounted_faces, Last_Encountered_faces, Frequency_faces], axis=1)

places_dummied = pd.concat([places_upsampled, First_encounted_places, Last_Encountered_places, Frequency_places], axis=1)

# %% Now to get the dummy variables for everything - THIS IS FOR DOWNSAMPLING

# First_encounted_faces = pd.get_dummies(faces_downsampled["First_encountered"], drop_first=False, prefix="FE")
# Last_Encountered_faces = pd.get_dummies(faces_downsampled["Last_Encountered"], drop_first=False, prefix="LE")
# Frequency_faces = pd.get_dummies(faces_downsampled["Frequency"], drop_first=False, prefix="FREQ")

# First_encounted_places = pd.get_dummies(places_downsampled["First_encountered"], drop_first=False, prefix="FE")
# Last_Encountered_places = pd.get_dummies(places_downsampled["Last_Encountered"], drop_first=False, prefix="LE")
# Frequency_places = pd.get_dummies(places_downsampled["Frequency"], drop_first=False, prefix="FREQ")

# %% Drop columns that are not needed anymore
# faces_downsampled.drop(["First_encountered", "Last_Encountered", "Frequency"], axis=1, inplace=True)

# places_downsampled.drop(["First_encountered", "Last_Encountered", "Frequency"], axis=1, inplace=True)

# %% Concatenate the dataframe with the new columns that contain the dummy variables
# faces_dummied = pd.concat([faces_downsampled, First_encounted_faces, Last_Encountered_faces, Frequency_faces], axis=1)

# places_dummied = pd.concat([places_downsampled, First_encounted_places, Last_Encountered_places, Frequency_places], axis=1)

# %% need to get labels

labels_F = pd.DataFrame(faces_dummied['Code'])
labels_F.Code[labels_F.Code == 'TOT'] = 1
labels_F.Code[labels_F.Code == 'known'] = 0

faces_complete = faces_dummied.drop(["Subject", "Code"], axis=1)
faces_complete = faces_complete[['FE_<6',
                                 'FE_6-11',
                                 'FE_12-15',
                                 'FE_16-20',
                                 'FE_21+',
                                 'LE_1 week ago',
                                 'LE_1 month ago',
                                 'LE_6 months ago',
                                 'LE_1 year ago',
                                 'LE_more than 2 years',
                                 'FREQ_1-50 times',
                                 'FREQ_51-100 times',
                                 'FREQ_101-150 times',
                                 'FREQ_151-200 times',
                                 'FREQ_200+times']]

faces_complete = faces_complete.apply(pd.to_numeric)
faces_complete = faces_complete.reset_index(drop=True)
labels_F = labels_F.apply(pd.to_numeric)
labels_F = labels_F.reset_index(drop=True)

labels_P = pd.DataFrame(places_dummied['Code'])
labels_P.Code[labels_P.Code == 'TOT'] = 1
labels_P.Code[labels_P.Code == 'known'] = 0

places_complete = places_dummied.drop(["Subject", "Code"], axis=1)
places_complete = places_complete[['FE_<6',
                                   'FE_6-11',
                                   'FE_12-15',
                                   'FE_16-20',
                                   'FE_21+',
                                   'LE_1 week ago',
                                   'LE_1 month ago',
                                   'LE_6 months ago',
                                   'LE_1 year ago',
                                   'LE_more than 2 years',
                                   'FREQ_1-50 times',
                                   'FREQ_51-100 times',
                                   'FREQ_101-150 times',
                                   'FREQ_151-200 times',
                                   'FREQ_200+times']]

places_complete = places_complete.apply(pd.to_numeric)
places_complete = places_complete.reset_index(drop=True)
labels_P = labels_P.apply(pd.to_numeric)
labels_P = labels_P.reset_index(drop=True)

# %% This is to run the pingouin logistic regression stats for everything together
y_F = labels_F["Code"]
X_F = faces_complete[['FE_<6',
                      'FE_6-11',
                      'FE_12-15',
                      'FE_16-20',
                      'FE_21+',
                      'LE_1 week ago',
                      'LE_1 month ago',
                      'LE_6 months ago',
                      'LE_1 year ago',
                      'LE_more than 2 years',
                      'FREQ_1-50 times',
                      'FREQ_51-100 times',
                      'FREQ_101-150 times',
                      'FREQ_151-200 times',
                      'FREQ_200+times']]

lom_face_complete = pg.logistic_regression(X_F, y_F, remove_na=True)
lom_face_complete = lom_face_complete.round(3)

y_P = labels_P["Code"]
X_P = places_complete[['FE_<6',
                       'FE_6-11',
                       'FE_12-15',
                       'FE_16-20',
                       'FE_21+',
                       'LE_1 week ago',
                       'LE_1 month ago',
                       'LE_6 months ago',
                       'LE_1 year ago',
                       'LE_more than 2 years',
                       'FREQ_1-50 times',
                       'FREQ_51-100 times',
                       'FREQ_101-150 times',
                       'FREQ_151-200 times',
                       'FREQ_200+times']]

lom_place_complete = pg.logistic_regression(X_P, y_P, remove_na=True)
lom_place_complete = lom_place_complete.round(3)

# %% This is to run the pingouin logistic regression stats for first encountered faces
y_F = labels_F["Code"]
X_F = faces_complete[['FE_<6', 'FE_6-11', 'FE_12-15', 'FE_16-20', 'FE_21+']]
lom_face_FE = pg.logistic_regression(X_F, y_F, remove_na=True)
lom_face_FE = lom_face_FE.round(3)

y_P = labels_P["Code"]
X_P = places_complete[['FE_<6', 'FE_6-11', 'FE_12-15', 'FE_16-20', 'FE_21+']]
lom_place_FE = pg.logistic_regression(X_P, y_P, remove_na=True)
lom_place_FE = lom_place_FE.round(3)

# %% This is to run the pingouin logistic regression stats for last encountered faces
y_F = labels_F["Code"]
X_F = faces_complete[['LE_1 week ago', 'LE_1 month ago', 'LE_6 months ago', 'LE_1 year ago', 'LE_more than 2 years']]
lom_face_LE = pg.logistic_regression(X_F, y_F, remove_na=True)
lom_face_LE = lom_face_LE.round(3)

y_P = labels_P["Code"]
X_P = places_complete[['LE_1 week ago', 'LE_1 month ago', 'LE_6 months ago', 'LE_1 year ago', 'LE_more than 2 years']]
lom_place_LE = pg.logistic_regression(X_P, y_P, remove_na=True)
lom_place_LE = lom_place_LE.round(3)

# %% This is to run the pingouin logistic regression stats for frequency faces
y_F = labels_F["Code"]
X_F = faces_complete[['FREQ_1-50 times', 'FREQ_51-100 times', 'FREQ_101-150 times', 'FREQ_151-200 times', 'FREQ_200+times']]
lom_face_FREQ = pg.logistic_regression(X_F, y_F, remove_na=True)
lom_face_FREQ = lom_face_FREQ.round(3)

y_P = labels_P["Code"]
X_P = places_complete[['FREQ_1-50 times', 'FREQ_51-100 times', 'FREQ_101-150 times', 'FREQ_151-200 times', 'FREQ_200+times']]
lom_place_FREQ = pg.logistic_regression(X_P, y_P, remove_na=True)
lom_place_FREQ = lom_place_FREQ.round(3)


# %% Data preparation of training and testing data sets for complete answers

x_train_F, x_test_F, y_train_F, y_test_F = train_test_split(faces_complete, labels_F, test_size=0.25, random_state=0)

x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(places_complete, labels_P, test_size=0.25, random_state=0)

# %% Time for the logistic regression

# fitting the model
logmodel_complete_F = LogisticRegression(class_weight="balanced")
# class_weight = balanced is a helpful argument for when the number of rejects is much higher than the number of acceptances adding
# to get classification up. Also helps in this case due to the difference between number of known and TOT responses

logmodel_complete_F.fit(x_train_F, y_train_F)
predictions_log_F = logmodel_complete_F.predict(x_test_F)

# fitting the model
logmodel_complete_P = LogisticRegression(class_weight="balanced")
# class_weight = balanced is a helpful argument for when the number of rejects is much higher than the number of acceptances adding
# to get classification up. Also helps in this case due to the difference between number of known and TOT responses

logmodel_complete_P.fit(x_train_P, y_train_P)
predictions_log_P = logmodel_complete_P.predict(x_test_P)

# %% Model evaluation

# Checking the precisions

precision_complete_F = (classification_report(y_test_F, predictions_log_F))

precision_complete_P = (classification_report(y_test_P, predictions_log_P))

# %% Feature importance

importance_complete_F = logmodel_complete_F.coef_.flatten()

importance_complete_P = logmodel_complete_P.coef_.flatten()

# %% Make a plot to show the features importance to the final  - PLACES

with sns.plotting_context("notebook", font_scale=0.9):
    pyplot.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams['figure.dpi'] = 1000
    pyplot.barh(places_complete.columns, importance_complete_P, color='indigo')
    pyplot.title("Barplot summary of importance of all answers - FACES")
    plt.xlim([-1.75, 1.75])
    plt.grid(False)
    pyplot.xlabel("score")
    pyplot.show()

# %% ROC Curve
logit_roc_auc = roc_auc_score(y_test_P, logmodel_complete_P.predict(x_test_P))
fpr, tpr, thresholds = roc_curve(y_test_P, logmodel_complete_P.predict_proba(x_test_P)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(False)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for complete faces')
plt.legend(loc="lower right")
plt.show()

# %% Preparing data for first encountered

faces_FE = faces_complete[['FE_<6', 'FE_6-11', 'FE_12-15', 'FE_16-20', 'FE_21+']]
faces_FE = faces_FE.apply(pd.to_numeric)

places_FE = places_complete[['FE_<6', 'FE_6-11', 'FE_12-15', 'FE_16-20', 'FE_21+']]
places_FE = places_FE.apply(pd.to_numeric)

# %% Data preparation of training and testing data sets for first encountered answers

x_train_F, x_test_F, y_train_F, y_test_F = train_test_split(faces_FE, labels_F, test_size=0.25, random_state=0)

x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(places_FE, labels_P, test_size=0.25, random_state=0)

# %% Time for the logistic regression

# fitting the model
logmodel_FE_F = LogisticRegression(class_weight="balanced")
# class_weight = balanced is a helpful argument for when the number of rejects is much higher than the number of acceptances adding
# to get classification up. Also helps in this case due to the difference between number of known and TOT responses

logmodel_FE_F.fit(x_train_F, y_train_F)
predictions_log_F = logmodel_FE_F.predict(x_test_F)

# fitting the model
logmodel_FE_P = LogisticRegression(class_weight="balanced")
# class_weight = balanced is a helpful argument for when the number of rejects is much higher than the number of acceptances adding
# to get classification up. Also helps in this case due to the difference between number of known and TOT responses

logmodel_FE_P.fit(x_train_P, y_train_P)
predictions_log_P = logmodel_FE_P.predict(x_test_P)
# %% Model evaluation

# Checking the precisions

precision_FE_F = (classification_report(y_test_F, predictions_log_F))

precision_FE_P = (classification_report(y_test_P, predictions_log_P))

# %% Feature importance

importance_FE_F = logmodel_FE_F.coef_.flatten()

importance_FE_P = logmodel_FE_P.coef_.flatten()

# %% Make a plot to show the features importance to the final data
with sns.plotting_context("notebook", font_scale=0.9):

    pyplot.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams['figure.dpi'] = 1000
    pyplot.barh(faces_FE.columns, importance_FE_F, color='indigo')
    pyplot.title("Barplot summary of importance first encountered answers")
    plt.xlim([-1.75, 1.75])
    plt.grid(False)
    pyplot.xlabel("score")
    pyplot.show()

# %% ROC Curve
logit_roc_auc = roc_auc_score(y_test_F, logmodel_FE_F.predict(x_test_F))
fpr, tpr, thresholds = roc_curve(y_test_F, logmodel_FE_F.predict_proba(x_test_F)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(False)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for "First encountered" faces')
plt.legend(loc="lower right")
plt.show()


# %% Preparing data for first encountered

faces_LE = faces_complete[['LE_1 week ago', 'LE_1 month ago', 'LE_6 months ago', 'LE_1 year ago', 'LE_more than 2 years']]
faces_LE = faces_LE.apply(pd.to_numeric)

places_LE = places_complete[['LE_1 week ago', 'LE_1 month ago', 'LE_6 months ago', 'LE_1 year ago', 'LE_more than 2 years']]
places_LE = places_LE.apply(pd.to_numeric)

# %% Data preparation of training and testing data sets for first encountered answers

x_train_F, x_test_F, y_train_F, y_test_F = train_test_split(faces_LE, labels_F, test_size=0.25, random_state=0)

x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(places_LE, labels_P, test_size=0.25, random_state=0)

# %% Time for the logistic regression

# fitting the model
logmodel_LE_F = LogisticRegression(class_weight="balanced")
# class_weight = balanced is a helpful argument for when the number of rejects is much higher than the number of acceptances adding
# to get classification up. Also helps in this case due to the difference between number of known and TOT responses

logmodel_LE_F.fit(x_train_F, y_train_F)
predictions_log_F = logmodel_LE_F.predict(x_test_F)

# fitting the model
logmodel_LE_P = LogisticRegression(class_weight="balanced")
# class_weight = balanced is a helpful argument for when the number of rejects is much higher than the number of acceptances adding
# to get classification up. Also helps in this case due to the difference between number of known and TOT responses

logmodel_LE_P.fit(x_train_P, y_train_P)
predictions_log_P = logmodel_LE_P.predict(x_test_P)

# %% Model evaluation

# Checking the precisions

precision_LE_F = (classification_report(y_test_F, predictions_log_F))

precision_LE_P = (classification_report(y_test_P, predictions_log_P))

# %% Feature importance

importance_LE_F = logmodel_LE_F.coef_.flatten()

importance_LE_P = logmodel_LE_P.coef_.flatten()

# %% Make a plot to show the features importance to the final data

with sns.plotting_context("notebook", font_scale=0.9):
    pyplot.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams['figure.dpi'] = 1000
    pyplot.barh(faces_LE.columns, importance_LE_F, color='indigo')
    pyplot.title("Barplot summary of importance of last encountered")
    plt.xlim([-1.75, 1.75])
    plt.grid(False)
    pyplot.xlabel("score")
    pyplot.show()

# %% ROC Curve
logit_roc_auc = roc_auc_score(y_test_F, logmodel_LE_F.predict(x_test_F))
fpr, tpr, thresholds = roc_curve(y_test_F, logmodel_LE_F.predict_proba(x_test_F)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(False)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for "Last encountered" faces')
plt.legend(loc="lower right")
plt.show()

# %% Preparing data for first encountered

faces_FREQ = faces_complete[['FREQ_1-50 times', 'FREQ_51-100 times', 'FREQ_101-150 times', 'FREQ_151-200 times', 'FREQ_200+times']]
faces_FREQ = faces_FREQ.apply(pd.to_numeric)

places_FREQ = places_complete[['FREQ_1-50 times', 'FREQ_51-100 times', 'FREQ_101-150 times', 'FREQ_151-200 times', 'FREQ_200+times']]
places_FREQ = places_FREQ.apply(pd.to_numeric)
# %% Data preparation of training and testing data sets for first encountered answers

x_train_F, x_test_F, y_train_F, y_test_F = train_test_split(faces_FREQ, labels_F, test_size=0.25, random_state=0)

x_train_P, x_test_P, y_train_P, y_test_P = train_test_split(places_FREQ, labels_P, test_size=0.25, random_state=0)

# %% Time for the logistic regression

# fitting the model
logmodel_FREQ_F = LogisticRegression(class_weight="balanced")
# class_weight = balanced is a helpful argument for when the number of rejects is much higher than the number of acceptances adding
# to get classification up. Also helps in this case due to the difference between number of known and TOT responses. However, this
# this is probably redundant now as the responses have been oversampled to make them equal

logmodel_FREQ_F.fit(x_train_F, y_train_F)
predictions_log_F = logmodel_FREQ_F.predict(x_test_F)

# fitting the model
logmodel_FREQ_P = LogisticRegression(class_weight="balanced")
# class_weight = balanced is a helpful argument for when the number of rejects is much higher than the number of acceptances adding
# to get classification up. Also helps in this case due to the difference between number of known and TOT responses. However, this
# this is probably redundant now as the responses have been oversampled to make them equal

logmodel_FREQ_P.fit(x_train_P, y_train_P)
predictions_log_P = logmodel_FREQ_P.predict(x_test_P)

# %% Model evaluation

# Checking the precisions

precision_FREQ_F = (classification_report(y_test_F, predictions_log_F))

precision_FREQ_P = (classification_report(y_test_P, predictions_log_P))

# %% Feature importance

importance_FREQ_F = logmodel_FREQ_F.coef_.flatten()

importance_FREQ_P = logmodel_FREQ_P.coef_.flatten()

# %% Make a plot to show the features importance to the final data
with sns.plotting_context("notebook", font_scale=0.9):
    pyplot.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams['figure.dpi'] = 1000
    pyplot.barh(faces_FREQ.columns, importance_FREQ_F, color='indigo')
    pyplot.title("Barplot summary of importance of frequency answers")
    plt.xlim([-1.75, 1.75])
    plt.grid(False)
    pyplot.xlabel("score")
    pyplot.show()

# %% ROC Curve
logit_roc_auc = roc_auc_score(y_test_F, logmodel_FREQ_F.predict(x_test_F))
fpr, tpr, thresholds = roc_curve(y_test_F, logmodel_FREQ_F.predict_proba(x_test_F)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(False)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for "Frequency" faces')
plt.legend(loc="lower right")
plt.show()
