# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
import os
import warnings

warnings.filterwarnings('ignore')

# gets the file name for the dataset
files = os.listdir('.')
dataset = files[0]

# loads the dataset
df = pd.read_csv(dataset, header=None)

# renames the column names
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
print(df.head())
print()

# checks the frequency distribution
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
for col in col_names:
    print(df[col].value_counts()) 
print()

# checks if there are missing values in the variables
df['class'].value_counts()
print(df.isnull().sum())

# target variables
X = df.drop(['class'], axis=1)
y = df['class']

# split data into separate training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# check the shape of X_train and X_test
X_train.shape, X_test.shape


