# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:20:19 2017

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "C:\\Users\\Admin\\Desktop\\PythonDataSets\\"
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')


#Pclass vs Survived
train[['Pclass','Survived']].groupby('Pclass').mean()

#Sex vs Survived
train[['Sex','Survived']].groupby('Sex').mean()

#Let's fill the age randomly beyween mean+- std
fullData = [train, test]

for data in fullData:
    mean = data['Age'].mean()
    std = data['Age'].std()
    nullCount = data['Age'].isnull().sum()
    
    data['Age'][np.isnan(data['Age'])]= np.random.randint(mean-std, mean+std, nullCount)

#SibSp + Parch
for data in fullData:
    data['FamilySize']= data['SibSp'] + data['Parch'] + 1

#Embarked
from statistics import mode
for data in fullData:
    data['Embarked'][data['Embarked'].isnull()] = mode(data['Embarked'])

#Cabin
for data in fullData:
    data.drop('Cabin', axis=1, inplace=True)

test.dropna(inplace=True)
#Creating Categorical Fare
for data in fullData:
    data['CategoricalFare'] = pd.qcut(data['Fare'], 4)
    data['CategoricalFare']=data['CategoricalFare'].astype(object)
    
    data.loc[ data['Fare'] <= 7.91, 'CategoricalFare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'CategoricalFare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'CategoricalFare'] = 2
    data.loc[ data['Fare'] > 31, 'CategoricalFare'] = 3
    data['CategoricalFare'] = data['CategoricalFare'].astype(int)
    
    
    data['Sex'] = data['Sex'].map({'male':1, 'female':0})
    data['Embarked'] = data['Embarked'].map({'C':0, 'Q':1, 'S':2})   

#Removing unwanted variables
CateogiesToRemove = ['PassengerId','Name', 'SibSp', 'Parch', 'Ticket','Fare']
for data in fullData:
    data.drop(CateogiesToRemove, axis=1, inplace=True)
    
X = train.drop('Survived', axis=1)
y = train['Survived']

from sklearn.cross_validation import StratifiedShuffleSplit,train_test_split
'''
sss = StratifiedShuffleSplit(y)

for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y)

#Building Models

#Logistic Regression
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score, precision_score,cohen_kappa_score

logReg_params = {'penalty':['l2'],
                 'C':[0.01,0.1,1],
                 'solver':['liblinear','newton-cg']}

logReg = LogisticRegression()
gs = GridSearchCV(logReg, param_grid=logReg_params, cv=10, verbose=3)

gs.fit(X_train, y_train)
pred_logReg = gs.predict(X_test)

#Calculating Metrics
Accuracy_logReg = accuracy_score(y_test, pred_logReg)
ROC_AUC_Score_logReg = roc_auc_score(y_test, pred_logReg)
Recall_logReg = recall_score(y_test, pred_logReg)
Kappa_logReg = cohen_kappa_score(y_test, pred_logReg)
Precision_logReg = precision_score(y_test, pred_logReg)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_params = {'n_estimators':[100],
             'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9],
              'max_features': [0.1]
              }

gs = GridSearchCV(rf, param_grid=rf_params, verbose=3)
gs.fit(X_train, y_train)
pred_rf = gs.predict(X_test)

Accuracy_rf = accuracy_score(y_test, pred_rf)
ROC_AUC_Score_rf = roc_auc_score(y_test, pred_rf)
Recall_rf = recall_score(y_test, pred_rf)
Kappa_rf = cohen_kappa_score(y_test, pred_rf)
Precision_rf = precision_score(y_test, pred_rf)


#chooses the best parameters for the classifier and outputs the evaluation metrics
def bestClassifier(clf, params):
    clfName = clf.__class__.__name__
    gs = GridSearchCV(clf, param_grid=params, verbose=0)
    gs.fit(X_train, y_train)
    pred = gs.predict(X_test)
    
    Accuracy = accuracy_score(y_test, pred)
    
    ROC_AUC_Score= roc_auc_score(y_test, pred)
    Recall = recall_score(y_test, pred)
    Kappa = cohen_kappa_score(y_test, pred)
    Precision = precision_score(y_test, pred)
    
    plt.plot(range(5), [Accuracy,ROC_AUC_Score,Recall,Kappa,Precision], marker='o')
    plt.xticks(range(5),['Accuracy','ROC_AUC_Score','Recall','Kappa','Precision'])
    plt.title("Scores for "+clfName)
    print('\n')
    
    print('********************************************')
    print('Best GridSearch Score: ', gs.best_score_)
    print('********************************************')
    print('\n')
    
    print('********************************************')
    print('Best Parameters for '+ clfName)
    print('********************************************')
    print(gs.best_params_)
    print('\n')
    
    print('************************************')
    print('Performance of '+ clfName)
    print('************************************')
    #print('\n')
    print('Accuracy: '+ str(Accuracy))
    print('ROC_AUC_Score: '+ str(ROC_AUC_Score))
    print('Recall: '+ str(Recall))
    print('Kappa: '+ str(Kappa))
    print('Precision: '+ str(Precision))
    print('\n')    
    
    