# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:54:41 2022

@author: jojot
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
import os
import warnings
import csv
import random

warnings.filterwarnings('ignore')

def findingFile(number):
    # gets the file name for the dataset from the folder
    files = os.listdir('.')
    dataset = files[number]
    return dataset

def randomHalfCSV(): # generates another csv file with about half the data picked randomly from the original csv
    total = 0
    maxim = 1728 // 2
    data = findingFile()
    file = open(data, 'r')
    rand = open("randomData.csv", 'w')
    dataread = csv.reader(file)
    for row in dataread:
        gatekeeper = random.randint(0, 4)
        if(gatekeeper % 2 == 0 and total != maxim):
            writ = csv.writer(rand)
            writ.writerow((row))
            total += 1
        if(total >= maxim):
            break

    #chosen_row = random.choice(list(dataread))
    #rand.write(chosen_row)
    # make a new csv file for new dataset
        
    
    
        


def randomForestTesting(number):
    dataset = findingFile(number)
    
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
    print(X_train.shape, X_test.shape)
    print()
    
    # check data types in X_train
    print(X_train.dtypes)
    print()
    
    # import category encoders
    import category_encoders as ce
    
    # encode categorical variables with ordinal encoding
    encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    
    # import Random Forest classifier
    from sklearn.ensemble import RandomForestClassifier
    
    # instantiate the classifier 
    rfc = RandomForestClassifier(random_state=0)
    
    # fit the model
    rfc.fit(X_train, y_train)
    
    # Predict the Test set results
    y_pred = rfc.predict(X_test)
    
    # Check accuracy score 
    from sklearn.metrics import accuracy_score
    
    print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    print()
    
    # instantiate the classifier with n_estimators = 100
    rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
    
    # fit the model to the training set
    rfc_100.fit(X_train, y_train)
    
    # Predict on the test set results
    y_pred_100 = rfc_100.predict(X_test)
    
    # Check accuracy score 
    print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
    print()
    
    # create the classifier with n_estimators = 100
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    
    # fit the model to the training set
    clf.fit(X_train, y_train)
    
    # view the feature scores
    feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_scores
    
    
    # Creating a seaborn bar plot
    sns.barplot(x=feature_scores, y=feature_scores.index)
    
    # Add labels to the graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    
    # Add title to the graph
    plt.title("Visualizing Important Features")
    
    # Visualize the graph
    plt.show()
    
    # declare feature vector and target variable
    X = df.drop(['class', 'doors'], axis=1)
    y = df['class']
    
    # split data into training and testing sets
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    
    # encode categorical variables with ordinal encoding
    encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    
    # instantiate the classifier with n_estimators = 100
    clf = RandomForestClassifier(random_state=0)
    
    # fit the model to the training set
    clf.fit(X_train, y_train)
    
    # Predict on the test set results
    y_pred = clf.predict(X_test)
    
    # Check accuracy score 
    print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    
    # Print the Confusion Matrix and slice it into four pieces
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n\n', cm)
    print()
    
    # Print the classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    
    
randomForestTesting(0)

