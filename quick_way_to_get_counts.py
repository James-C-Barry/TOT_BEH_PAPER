# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:26:57 2023

@author: jericho
"""

import pandas as pd
import os
import numpy as np

os.chdir("C:\\Users\\jericho\\OneDrive\\Work Stuff\\TOT_BEH_Results")

counts = pd.read_csv("RESULTS_PLACES_MASTER.csv", sep=',')


countsTOT = counts[counts["Code"].str.contains("known") == False]
countsTOT = countsTOT[countsTOT["Code"].str.contains("familiar") == False]
countsTOT = countsTOT[countsTOT["Code"].str.contains("unknown") == False]
countsTOT = countsTOT.groupby(['Subject', 'Frequency']).size()

countsTOT_FREQ = countsTOT.to_frame()
countsTOT_FREQ.reset_index(inplace=True)
countsTOT_FREQ = countsTOT_FREQ.set_axis(['Subject', 'Code', 'Count'], axis=1, inplace=False)
countsTOT_FREQ = countsTOT_FREQ.pivot(index='Subject', columns='Code', values='Count')

countsTOT_FREQ = countsTOT_FREQ. replace(np. nan,0) 