#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:38:24 2022

@author: jbarry

Script to clean TOT Normative log files and produce the desired output
"""

#%% imports
import os
import pandas as pd

#%% set directory
os.chdir("/home/jbarry/jbarry/TOT_Norm_Results/places")





#%%Now starts the recognition part
list = []
TOTNorm = open("s_9074-TOT_Norm.log")
line = TOTNorm.readline()
while line:
    try:list.append(line)
    except ValueError:print('Error in line :' + line )
    line = TOTNorm.readline()
#%% split and name the coloumns 
TOTNorm = pd.DataFrame([sub.split("\t") for sub in list])
TOTNorm = TOTNorm.iloc[23:]
TOTNorm.columns = ["subject", "Trial", "Event Type", "Code", "Time", "TTime", "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur","None",]
TOTNorm.head()
TOTNorm 


#%% Cleaning Data and removing symbols and numbers before trial name

TOTNorm['Trial'] = TOTNorm['Trial'].str.replace('_','')
#post_fMRI_clean['Trial'] = post_fMRI_clean['Trial'].str.replace('_','')
TOTNorm['Trial'] = TOTNorm['Trial'].str.replace('\d+', '')
#post_fMRI_clean['Trial'] = post_fMRI_clean['Trial'].str.replace('\d+', '')

#%% remove unwanted columns and rows
TOTNorm.drop(["Time", "TTime", 'Event Type', "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur", "None"], inplace=True, axis=1)
TOTNorm = TOTNorm[TOTNorm["subject"].str.contains("9074") == False]
TOTNorm = TOTNorm.iloc[13:]
TOTNorm = TOTNorm.drop(TOTNorm[TOTNorm['Trial'].str.match('Recognition',na=False)].index) 
TOTNorm = TOTNorm.reset_index(drop=True)

#%% move all the responses up a column so they match the picture
TOTNorm['Code'] = TOTNorm['Code'].shift(-1)

#%% remove redundant row and column
TOTNorm = TOTNorm.drop(TOTNorm[TOTNorm['Trial'].str.match('Choice',na=False)].index)
TOTNorm.drop("subject", inplace=True, axis=1)

#%% Rename responses to match what they are
TOTNorm['Code'] = TOTNorm['Code'].replace({'1':'known'})
TOTNorm['Code'] = TOTNorm['Code'].replace({'2':'known'})
TOTNorm['Code'] = TOTNorm['Code'].replace({'6':'unknown'})
TOTNorm['Code'] = TOTNorm['Code'].replace({'7':'unknown'})

#%%
list = []
TOTNormRec = open("s_9074-TOT_Norm.log")
line = TOTNormRec.readline()
while line:
    try:list.append(line)
    except ValueError:print('Error in line :' + line )
    line = TOTNormRec.readline()
#%% split and name the coloumns 
TOTNormRec = pd.DataFrame([sub.split("\t") for sub in list])
TOTNormRec = TOTNormRec.iloc[23:]
TOTNormRec.columns = ["subject", "Trial", "Event Type", "Code", "Time", "TTime", "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur","None",]
TOTNormRec.head()
TOTNormRec 


#%% remove unwanted columns and rows
TOTNormRec.drop(["Time", "TTime", 'Event Type', "Uncertainty", "Duration", "Uncertainty", "ReqTime", "ReqDur", "None"], inplace=True, axis=1)
TOTNormRec = TOTNormRec[TOTNormRec["subject"].str.contains("9074") == False]
TOTNormRec = TOTNormRec.iloc[13:]
TOTNormRec = TOTNormRec.drop(TOTNormRec[TOTNormRec['Trial'].str.match('Choice',na=False)].index) 
TOTNormRec = TOTNormRec.reset_index(drop=True)

#%% move all the responses up a column so they match the picture
TOTNormRec['Code'] = TOTNormRec['Code'].shift(-1)

#%% remove redundant row and column
TOTNormRec = TOTNormRec.drop(TOTNormRec[TOTNormRec['Trial'].str.match('Recognition',na=False)].index)
TOTNormRec.drop("subject", inplace=True, axis=1)
TOTNormRec = TOTNormRec.reset_index(drop=True)

#%% Remove all numbers and characters before the first name
TOTNormRec['Trial'] = TOTNormRec['Trial'].str.strip("1, 2, 3, 4, 5, 6, 7, 8, 9, 0, _")

#%% Convert Code column into integers and minus 1 to get the correct value
TOTNormRec['Code'] = TOTNormRec['Code'].astype(int)
TOTNormRec.loc[:, "Code"] = TOTNormRec["Code"].apply(lambda x: x - 1)
TOTNormRec['Code'] = TOTNormRec['Code'].astype(str)

#%% insert new column
TOTNormRec.insert(2, "correct/incorrect", "")

#%%
TOTNormRec.loc[(TOTNormRec['Trial'].str.contains ('1')) & (TOTNormRec['Code'].str.contains('1')), 'correct/incorrect'] = 'correct'
TOTNormRec.loc[(TOTNormRec['Trial'].str.contains ('2')) & (TOTNormRec['Code'].str.contains('2')), 'correct/incorrect'] = 'correct'
TOTNormRec.loc[(TOTNormRec['Trial'].str.contains ('3')) & (TOTNormRec['Code'].str.contains('3')), 'correct/incorrect'] = 'correct'
TOTNormRec.loc[(TOTNormRec['Trial'].str.contains ('4')) & (TOTNormRec['Code'].str.contains('4')), 'correct/incorrect'] = 'correct'
TOTNormRec.loc[(TOTNormRec['Trial'].str.contains ('1')) & (TOTNormRec['Code'].str.contains('2|3|4|5|6')), 'correct/incorrect'] = 'incorrect'
TOTNormRec.loc[(TOTNormRec['Trial'].str.contains ('2')) & (TOTNormRec['Code'].str.contains('1|3|4|5|6')), 'correct/incorrect'] = 'incorrect'
TOTNormRec.loc[(TOTNormRec['Trial'].str.contains ('3')) & (TOTNormRec['Code'].str.contains('1|2|4|5|6')), 'correct/incorrect'] = 'incorrect'
TOTNormRec.loc[(TOTNormRec['Trial'].str.contains ('4')) & (TOTNormRec['Code'].str.contains('1|2|3|5|6')), 'correct/incorrect'] = 'incorrect'

#%% Move column to original
TOTNorm['correct/incorrect'] = TOTNormRec['correct/incorrect'].values

#%% Final Clean I think 
TOTNorm.loc[(TOTNorm['Code'].str.contains ('known')) & (TOTNorm['correct/incorrect'].str.contains ('incorrect')), 'Code'] = 'unknown'