#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:38:24 2022

@author: jbarry

Script to extract data from the behavioural study 
"""

#%% imports
import os
import pandas as pd

#%% set directory
os.chdir("/home/jbarry/jbarry/TOTBeh_RESULTS/places")





#%%Now starts the choice part
list = []
TOTBEH = open("s_9792-TOT_BEH_Place.log")
line = TOTBEH.readline()
while line:
    try:list.append(line)
    except ValueError:print('Error in line :' + line )
    line = TOTBEH.readline()
#%% split and name the coloumns 
TOTBEH = pd.DataFrame([sub.split("\t") for sub in list])
TOTBEH = TOTBEH.iloc[23:]
TOTBEH.columns = ["subject", "Trial", "Event Type", "Code", "Time", "TTime", "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur","Stim Type", "Pair Index"]
TOTBEH.head()
TOTBEH 


#%% Cleaning Data and removing symbols and numbers before trial name

TOTBEH['Trial'] = TOTBEH['Trial'].str.replace('_','')
#TOTBEH['Trial'] = TOTBEH['Trial'].str.replace('_','')
TOTBEH['Trial'] = TOTBEH['Trial'].str.replace('\d+', '')
#TOTBEH['Trial'] = TOTBEH['Trial'].str.replace('\d+', '')

#%% remove unwanted columns and rows
TOTBEH.drop(["TTime", 'Event Type', "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur", "Time","Stim Type", "Pair Index"], inplace=True, axis=1)
TOTBEH = TOTBEH[TOTBEH["subject"].str.contains("s_9792") == False]
TOTBEH = TOTBEH.iloc[8:]
TOTBEH = TOTBEH.drop(TOTBEH[TOTBEH['Trial'].str.match('Recognition',na=False)].index)
TOTBEH = TOTBEH.drop(TOTBEH[TOTBEH['Trial'].str.match('Firstencounter',na=False)].index) 
TOTBEH = TOTBEH.drop(TOTBEH[TOTBEH['Trial'].str.match('LastEncountered',na=False)].index) 
TOTBEH = TOTBEH.drop(TOTBEH[TOTBEH['Trial'].str.match('Frequency',na=False)].index)
TOTBEH = TOTBEH.drop(TOTBEH[TOTBEH['Trial'].str.match('Instructions',na=False)].index) 
TOTBEH = TOTBEH.reset_index(drop=True)

#%% move all the responses up a column so they match the picture
TOTBEH['Code'] = TOTBEH['Code'].shift(-1)

#%% Repalce possible NAN entry if last image is not recognised
TOTBEH = TOTBEH.fillna('')

#%% remove redundant row and column
TOTBEH = TOTBEH.drop(TOTBEH[TOTBEH['Trial'].str.match('Choice',na=False)].index)
TOTBEH.drop("subject", inplace=True, axis=1)

#%% Rename responses to match what they are
#TOTBEH['Code'] = TOTBEH['Code'].replace({'1':'known'})
#TOTBEH['Code'] = TOTBEH['Code'].replace({'2':'known'})
#TOTBEH['Code'] = TOTBEH['Code'].replace({'6':'unknown'})
#TOTBEH['Code'] = TOTBEH['Code'].replace({'7':'unknown'})

#%% Now for the Recognition part
list = []
TOTBEHRec = open("s_9792-TOT_BEH_Place.log")
line = TOTBEHRec.readline()
while line:
    try:list.append(line)
    except ValueError:print('Error in line :' + line )
    line = TOTBEHRec.readline()
#%% split and name the coloumns 
TOTBEHRec = pd.DataFrame([sub.split("\t") for sub in list])
TOTBEHRec = TOTBEHRec.iloc[23:]
TOTBEHRec.columns = ["subject", "Trial", "Event Type", "Code", "Time", "TTime", "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur","Stim Type", "Pair Index"]
TOTBEHRec.head()
TOTBEHRec 

#%% Remove all numbers and characters before the first name
TOTBEHRec['Trial'] = TOTBEHRec['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% remove unwanted columns and rows
TOTBEHRec.drop(["TTime", 'Event Type', "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur", "Time","Stim Type", "Pair Index"], inplace=True, axis=1)
TOTBEHRec = TOTBEHRec[TOTBEHRec["subject"].str.contains("s_9792") == False]
TOTBEHRec = TOTBEHRec.iloc[8:]
TOTBEHRec = TOTBEHRec.drop(TOTBEHRec[TOTBEHRec['Trial'].str.match('Choice',na=False)].index)
TOTBEHRec = TOTBEHRec.drop(TOTBEHRec[TOTBEHRec['Trial'].str.match('First_encounter',na=False)].index) 
TOTBEHRec = TOTBEHRec.drop(TOTBEHRec[TOTBEHRec['Trial'].str.match('Last_Encountered',na=False)].index) 
TOTBEHRec = TOTBEHRec.drop(TOTBEHRec[TOTBEHRec['Trial'].str.match('Frequency',na=False)].index)
TOTBEHRec = TOTBEHRec.drop(TOTBEHRec[TOTBEHRec['Trial'].str.match('Instructions',na=False)].index) 
TOTBEHRec = TOTBEHRec.reset_index(drop=True)

#%% move all the responses up a column so they match the picture
TOTBEHRec['Code'] = TOTBEHRec['Code'].shift(-1)

#%% Repalce possible NAN entry if last image is not recognised
TOTBEHRec = TOTBEHRec.fillna('')

#%% remove redundant row and column
TOTBEHRec = TOTBEHRec.drop(TOTBEHRec[TOTBEHRec['Trial'].str.match('Recognition',na=False)].index)
TOTBEHRec.drop("subject", inplace=True, axis=1)
TOTBEHRec = TOTBEHRec.reset_index(drop=True)

#%% Remove all numbers and characters before the first name
TOTBEHRec['Trial'] = TOTBEHRec['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% insert new column
TOTBEHRec.insert(2, "1=correct/2=incorrect", "")

#%%
TOTBEHRec.loc[(TOTBEHRec['Trial'].str.contains ('1')) & (TOTBEHRec['Code'].str.contains('1')), '1=correct/2=incorrect'] = '1'
TOTBEHRec.loc[(TOTBEHRec['Trial'].str.contains ('2')) & (TOTBEHRec['Code'].str.contains('2')), '1=correct/2=incorrect'] = '1'
TOTBEHRec.loc[(TOTBEHRec['Trial'].str.contains ('3')) & (TOTBEHRec['Code'].str.contains('3')), '1=correct/2=incorrect'] = '1'
TOTBEHRec.loc[(TOTBEHRec['Trial'].str.contains ('4')) & (TOTBEHRec['Code'].str.contains('4')), '1=correct/2=incorrect'] = '1'
TOTBEHRec.loc[(TOTBEHRec['Trial'].str.contains ('1')) & (TOTBEHRec['Code'].str.contains('2|3|4|5|6')), '1=correct/2=incorrect'] = '2'
TOTBEHRec.loc[(TOTBEHRec['Trial'].str.contains ('2')) & (TOTBEHRec['Code'].str.contains('1|3|4|5|6')), '1=correct/2=incorrect'] = '2'
TOTBEHRec.loc[(TOTBEHRec['Trial'].str.contains ('3')) & (TOTBEHRec['Code'].str.contains('1|2|4|5|6')), '1=correct/2=incorrect'] = '2'
TOTBEHRec.loc[(TOTBEHRec['Trial'].str.contains ('4')) & (TOTBEHRec['Code'].str.contains('1|2|3|5|6')), '1=correct/2=incorrect'] = '2'

#%% Move column to original
TOTBEH['1=correct/2=incorrect'] = TOTBEHRec['1=correct/2=incorrect'].values

#%% Now for the First Encountered part
list = []
TOTBEHFirstEncounter = open("s_9792-TOT_BEH_Place.log")
line = TOTBEHFirstEncounter.readline()
while line:
    try:list.append(line)
    except ValueError:print('Error in line :' + line )
    line = TOTBEHFirstEncounter.readline()
#%% split and name the coloumns 
TOTBEHFirstEncounter = pd.DataFrame([sub.split("\t") for sub in list])
TOTBEHFirstEncounter = TOTBEHFirstEncounter.iloc[23:]
TOTBEHFirstEncounter.columns = ["subject", "Trial", "Event Type", "Code", "Time", "TTime", "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur","Stim Type", "Pair Index"]
TOTBEHFirstEncounter.head()
TOTBEHFirstEncounter 

#%% Remove all numbers and characters before the first name
TOTBEHFirstEncounter['Trial'] = TOTBEHFirstEncounter['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% remove unwanted columns and rows
TOTBEHFirstEncounter.drop(["TTime", 'Event Type', "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur", "Time","Stim Type", "Pair Index"], inplace=True, axis=1)
TOTBEHFirstEncounter = TOTBEHFirstEncounter[TOTBEHFirstEncounter["subject"].str.contains("s_9792") == False]
TOTBEHFirstEncounter = TOTBEHFirstEncounter.iloc[8:]
TOTBEHFirstEncounter = TOTBEHFirstEncounter.drop(TOTBEHFirstEncounter[TOTBEHFirstEncounter['Trial'].str.match('Choice',na=False)].index)
TOTBEHFirstEncounter = TOTBEHFirstEncounter.drop(TOTBEHFirstEncounter[TOTBEHFirstEncounter['Trial'].str.match('Recognition',na=False)].index) 
TOTBEHFirstEncounter = TOTBEHFirstEncounter.drop(TOTBEHFirstEncounter[TOTBEHFirstEncounter['Trial'].str.match('Last_Encountered',na=False)].index) 
TOTBEHFirstEncounter = TOTBEHFirstEncounter.drop(TOTBEHFirstEncounter[TOTBEHFirstEncounter['Trial'].str.match('Frequency',na=False)].index)
TOTBEHFirstEncounter = TOTBEHFirstEncounter.drop(TOTBEHFirstEncounter[TOTBEHFirstEncounter['Trial'].str.match('Instructions',na=False)].index) 
TOTBEHFirstEncounter = TOTBEHFirstEncounter.reset_index(drop=True)

#%% move all the responses up a column so they match the picture
TOTBEHFirstEncounter['Code'] = TOTBEHFirstEncounter['Code'].shift(-1)

#%% Repalce possible NAN entry if last image is not recognised
TOTBEHFirstEncounter = TOTBEHFirstEncounter.fillna('')

#%% remove redundant row and column
TOTBEHFirstEncounter = TOTBEHFirstEncounter.drop(TOTBEHFirstEncounter[TOTBEHFirstEncounter['Trial'].str.match('First_encounter',na=False)].index)
TOTBEHFirstEncounter.drop("subject", inplace=True, axis=1)
TOTBEHFirstEncounter = TOTBEHFirstEncounter.reset_index(drop=True)

#%% Remove all numbers and characters before the first name
TOTBEHFirstEncounter['Trial'] = TOTBEHFirstEncounter['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% insert new column
TOTBEHFirstEncounter.insert(2, "First_encountered", "")

#%%
TOTBEHFirstEncounter.loc[(TOTBEHFirstEncounter['Code'].str.contains ('1')), 'First_encountered'] = '1'
TOTBEHFirstEncounter.loc[(TOTBEHFirstEncounter['Code'].str.contains ('2')), 'First_encountered'] = '2'
TOTBEHFirstEncounter.loc[(TOTBEHFirstEncounter['Code'].str.contains ('3')), 'First_encountered'] = '3'
TOTBEHFirstEncounter.loc[(TOTBEHFirstEncounter['Code'].str.contains ('4')), 'First_encountered'] = '4'
TOTBEHFirstEncounter.loc[(TOTBEHFirstEncounter['Code'].str.contains ('5')), 'First_encountered'] = '5'


#%% Move column to original
TOTBEH['First_encountered'] = TOTBEHFirstEncounter['First_encountered'].values

#%% Now for the Last Encountered part
list = []
TOTBEHLastEncounter = open("s_9792-TOT_BEH_Place.log")
line = TOTBEHLastEncounter.readline()
while line:
    try:list.append(line)
    except ValueError:print('Error in line :' + line )
    line = TOTBEHLastEncounter.readline()
#%% split and name the coloumns 
TOTBEHLastEncounter = pd.DataFrame([sub.split("\t") for sub in list])
TOTBEHLastEncounter = TOTBEHLastEncounter.iloc[23:]
TOTBEHLastEncounter.columns = ["subject", "Trial", "Event Type", "Code", "Time", "TTime", "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur","Stim Type", "Pair Index"]
TOTBEHLastEncounter.head()
TOTBEHLastEncounter 

#%% Remove all numbers and characters before the first name
TOTBEHLastEncounter['Trial'] = TOTBEHLastEncounter['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% remove unwanted columns and rows
TOTBEHLastEncounter.drop(["TTime", 'Event Type', "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur", "Time","Stim Type", "Pair Index"], inplace=True, axis=1)
TOTBEHLastEncounter = TOTBEHLastEncounter[TOTBEHLastEncounter["subject"].str.contains("s_9792") == False]
TOTBEHLastEncounter = TOTBEHLastEncounter.iloc[8:]
TOTBEHLastEncounter = TOTBEHLastEncounter.drop(TOTBEHLastEncounter[TOTBEHLastEncounter['Trial'].str.match('Choice',na=False)].index)
TOTBEHLastEncounter = TOTBEHLastEncounter.drop(TOTBEHLastEncounter[TOTBEHLastEncounter['Trial'].str.match('Recognition',na=False)].index) 
TOTBEHLastEncounter = TOTBEHLastEncounter.drop(TOTBEHLastEncounter[TOTBEHLastEncounter['Trial'].str.match('First_encounter',na=False)].index) 
TOTBEHLastEncounter = TOTBEHLastEncounter.drop(TOTBEHLastEncounter[TOTBEHLastEncounter['Trial'].str.match('Frequency',na=False)].index)
TOTBEHLastEncounter = TOTBEHLastEncounter.drop(TOTBEHLastEncounter[TOTBEHLastEncounter['Trial'].str.match('Instructions',na=False)].index) 
TOTBEHLastEncounter = TOTBEHLastEncounter.reset_index(drop=True)

#%% move all the responses up a column so they match the picture
TOTBEHLastEncounter['Code'] = TOTBEHLastEncounter['Code'].shift(-1)

#%% Repalce possible NAN entry if last image is not recognised
TOTBEHLastEncounter = TOTBEHLastEncounter.fillna('')

#%% remove redundant row and column
TOTBEHLastEncounter = TOTBEHLastEncounter.drop(TOTBEHLastEncounter[TOTBEHLastEncounter['Trial'].str.match('Last_Encountered',na=False)].index)
TOTBEHLastEncounter.drop("subject", inplace=True, axis=1)
TOTBEHLastEncounter = TOTBEHLastEncounter.reset_index(drop=True)

#%% Remove all numbers and characters before the first name
TOTBEHLastEncounter['Trial'] = TOTBEHLastEncounter['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% insert new column
TOTBEHLastEncounter.insert(2, "Last_Encountered", "")

#%%
TOTBEHLastEncounter.loc[(TOTBEHLastEncounter['Code'].str.contains ('1')), 'Last_Encountered'] = '1'
TOTBEHLastEncounter.loc[(TOTBEHLastEncounter['Code'].str.contains ('2')), 'Last_Encountered'] = '2'
TOTBEHLastEncounter.loc[(TOTBEHLastEncounter['Code'].str.contains ('3')), 'Last_Encountered'] = '3'
TOTBEHLastEncounter.loc[(TOTBEHLastEncounter['Code'].str.contains ('4')), 'Last_Encountered'] = '4'
TOTBEHLastEncounter.loc[(TOTBEHLastEncounter['Code'].str.contains ('5')), 'Last_Encountered'] = '5'

#%% Move column to original
TOTBEH['Last_Encountered'] = TOTBEHLastEncounter['Last_Encountered'].values

#%% Now for the frequency part
list = []
TOTBEHFrequency = open("s_9792-TOT_BEH_Place.log")
line = TOTBEHFrequency.readline()
while line:
    try:list.append(line)
    except ValueError:print('Error in line :' + line )
    line = TOTBEHFrequency.readline()
#%% split and name the coloumns 
TOTBEHFrequency = pd.DataFrame([sub.split("\t") for sub in list])
TOTBEHFrequency = TOTBEHFrequency.iloc[23:]
TOTBEHFrequency.columns = ["subject", "Trial", "Event Type", "Code", "Time", "TTime", "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur","Stim Type", "Pair Index"]
TOTBEHFrequency.head()
TOTBEHFrequency 

#%% Remove all numbers and characters before the first name
TOTBEHFrequency['Trial'] = TOTBEHFrequency['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% remove unwanted columns and rows
TOTBEHFrequency.drop(["TTime", 'Event Type', "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur", "Time","Stim Type", "Pair Index"], inplace=True, axis=1)
TOTBEHFrequency = TOTBEHFrequency[TOTBEHFrequency["subject"].str.contains("s_9792") == False]
TOTBEHFrequency = TOTBEHFrequency.iloc[8:]
TOTBEHFrequency = TOTBEHFrequency.drop(TOTBEHFrequency[TOTBEHFrequency['Trial'].str.match('Choice',na=False)].index)
TOTBEHFrequency = TOTBEHFrequency.drop(TOTBEHFrequency[TOTBEHFrequency['Trial'].str.match('Recognition',na=False)].index) 
TOTBEHFrequency = TOTBEHFrequency.drop(TOTBEHFrequency[TOTBEHFrequency['Trial'].str.match('First_encounter',na=False)].index) 
TOTBEHFrequency = TOTBEHFrequency.drop(TOTBEHFrequency[TOTBEHFrequency['Trial'].str.match('Last_Encountered',na=False)].index)
TOTBEHFrequency = TOTBEHFrequency.drop(TOTBEHFrequency[TOTBEHFrequency['Trial'].str.match('Instructions',na=False)].index) 
TOTBEHFrequency = TOTBEHFrequency.reset_index(drop=True)

#%% move all the responses up a column so they match the picture
TOTBEHFrequency['Code'] = TOTBEHFrequency['Code'].shift(-1)

#%% Repalce possible NAN entry if last image is not recognised
TOTBEHFrequency = TOTBEHFrequency.fillna('')

#%% remove redundant row and column
TOTBEHFrequency = TOTBEHFrequency.drop(TOTBEHFrequency[TOTBEHFrequency['Trial'].str.match('Frequency',na=False)].index)
TOTBEHFrequency.drop("subject", inplace=True, axis=1)
TOTBEHFrequency = TOTBEHFrequency.reset_index(drop=True)

#%% Remove all numbers and characters before the first name
TOTBEHFrequency['Trial'] = TOTBEHFrequency['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% insert new column
TOTBEHFrequency.insert(2, "Frequency", "")

#%%
TOTBEHFrequency.loc[(TOTBEHFrequency['Code'].str.contains ('1')), 'Frequency'] = '1'
TOTBEHFrequency.loc[(TOTBEHFrequency['Code'].str.contains ('2')), 'Frequency'] = '2'
TOTBEHFrequency.loc[(TOTBEHFrequency['Code'].str.contains ('3')), 'Frequency'] = '3'
TOTBEHFrequency.loc[(TOTBEHFrequency['Code'].str.contains ('4')), 'Frequency'] = '4'
TOTBEHFrequency.loc[(TOTBEHFrequency['Code'].str.contains ('5')), 'Frequency'] = '5'

#%% Move column to original
TOTBEH['Frequency'] = TOTBEHFrequency['Frequency'].values

#%% Rename responses to match what they are
TOTBEH['Code'] = TOTBEH['Code'].replace({'1':'known'})
TOTBEH['Code'] = TOTBEH['Code'].replace({'2':'TOT'})
TOTBEH['Code'] = TOTBEH['Code'].replace({'3':'familiar'})
TOTBEH['Code'] = TOTBEH['Code'].replace({'4':'unknown'})
TOTBEH['Code'] = TOTBEH['Code'].replace({'5':'unknown'})
TOTBEH['Code'] = TOTBEH['Code'].replace({'6':'unknown'})

#%% Rename responses to match what they are
TOTBEH['1=correct/2=incorrect'] = TOTBEH['1=correct/2=incorrect'].replace({'1':'correct'})
TOTBEH['1=correct/2=incorrect'] = TOTBEH['1=correct/2=incorrect'].replace({'2':'incorrect'})
