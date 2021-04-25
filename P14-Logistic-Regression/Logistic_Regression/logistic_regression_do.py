# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:13:08 2020

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.3, random_state = 0)
