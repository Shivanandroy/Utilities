# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 18:23:47 2017

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target

data = pd.DataFrame(X, columns=boston.feature_names)
features = boston.feature_names.tolist()

#plotting the predictors against the target

import itertools #importing itertools
fig=plt.subplots(figsize=(15,30))#setting the overall figure size
length=X.shape[1] #number of predictors
for col, i in itertools.zip_longest(features,range(length)):
    #plotting 2 subplots in each row
    plt.subplot(np.ceil(length/2),2,i+1)
    plt.subplots_adjust(hspace=.5)
    plt.scatter(data[col],y)
    plt.xticks(rotation=90)
    plt.xlabel(col)
    plt.ylabel('Prices')
    
#plotting 1 subplot at a time
plt.subplots(figsize=(10,100))
for col, i in itertools.zip_longest(features, range(length)):
    plt.subplot(length,1,i+1)
    plt.scatter(data[col],y)
    plt.xlabel(col)
    plt.ylabel('Price')
    

    