#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:13:53 2022

@author: jbarry

Purpose: This is to extract the RT times from the CSV files and then add them to the final recoded file. This can be done without adding
just don't run the last cell
"""

import pandas as pd
import os
import numpy as np

os.chdir("C:\\Users\\jericho\\OneDrive\\Work Stuff\\TOT_BEH_Results\\places")

names = ["Subject", "Trial", "Event Type", "Code", "Time", "TTime", "Uncertainty", "Duration", "Uncertainty2", "ReqTime", "ReqDur","Stim Type", "Pair Index"]

RT_Time = pd.read_csv("s_9624-TOT_BEH_Place.log", sep='\t', names = names)


#%% Clean data
RT_Time.drop(['Event Type', "Duration","Uncertainty","Code", "ReqTime", "TTime", "ReqDur", "Stim Type", "Pair Index"], inplace=True, axis=1)
RT_Time = RT_Time[RT_Time["Subject"].str.contains("s_9624") == False]
RT_Time = RT_Time[RT_Time["Subject"].str.contains("Scenario") == False]
RT_Time = RT_Time[RT_Time["Subject"].str.contains("Logfile") == False]
RT_Time = RT_Time[RT_Time["Subject"].str.contains("Subject") == False]
RT_Time = RT_Time[RT_Time["Subject"].str.contains("Event Type") == False]
RT_Time = RT_Time.drop(RT_Time[RT_Time['Trial'].str.match('Recognition',na=False)].index)
RT_Time = RT_Time.drop(RT_Time[RT_Time['Trial'].str.match('First_encounter',na=False)].index) 
RT_Time = RT_Time.drop(RT_Time[RT_Time['Trial'].str.match('Last_Encountered',na=False)].index) 
RT_Time = RT_Time.drop(RT_Time[RT_Time['Trial'].str.match('Frequency',na=False)].index)
RT_Time = RT_Time.drop(RT_Time[RT_Time['Trial'].str.match('Instructions',na=False)].index) 
RT_Time = RT_Time.drop(RT_Time[RT_Time['Trial'].str.match('Response',na=False)].index) 
RT_Time = RT_Time.reset_index(drop=True)


#%% Move rows up
RT_Time['Time'] = RT_Time['Time'].shift(-1)
RT_Time["Time"] = RT_Time.Time.fillna(RT_Time.Uncertainty2)

#%% Add values from below
#RT_Time["TTime"] = pd.to_numeric(RT_Time["TTime"])
#RT_Time['RT'] = RT_Time['Duration'] + RT_Time['Duration'].shift(periods=-1, fill_value=0)

#%% Moving the values up

#idx_Choice=RT_Time.loc[RT_Time.Code.str.contains('Choice')]
#times = idx_Choice['Time']
#idx = idx_Choice.index
#idx = idx-2
#times = times.to_numpy()
#RT_Time.loc[idx, 'Time'] = times

#%% Cleaning Data and removing symbols and numbers before trial name

#RT_Time['Code'] = RT_Time['Code'].str.replace('_','')
RT_Time['Trial'] = RT_Time['Trial'].str.replace('_','')
#RT_Time['Code'] = RT_Time['Code'].str.replace('\d+', '')
RT_Time['Trial'] = RT_Time['Trial'].str.replace('\d+', '')


#%% Clean data
RT_Time = RT_Time.drop(RT_Time[RT_Time['Trial'].str.match('Choice',na=False)].index) 
RT_Time['Subject'] = RT_Time['Subject'].str.replace('Picture', 'S_9624')
#RT_Time = RT_Time.replace(r'^s*$', float('NaN'), regex = True)
#RT_Time.dropna(inplace = True) 
#%% rename columns
RT_Time = RT_Time.drop('Uncertainty2', axis = 1)

#%% make data numbers instead of strings
RT_Time["Time"] = pd.to_numeric(RT_Time["Time"])

#%% calculate the onset times
RT_Time["Time"] = RT_Time["Time"]/10000
#RT_Time["Time"] = RT_Time["Time"].astype(str)

