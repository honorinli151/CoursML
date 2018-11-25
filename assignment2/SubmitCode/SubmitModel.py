'''
File: f:\ML A2\SubmitModel.py
Project: f:\ML A2
Created Date: Sunday November 25th 2018
Author: Chenle Li
-----
Last Modified: 2018-11-25 10:33:10
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
'''
#%%
import numpy as np
import pandas as pd
import sklearn
import re
import seaborn as sns
try:
    import xgboost 
    from xgboost import XGBClassifier
except ImportError:
    print("Please install xgboost, refer to https://xgboost.readthedocs.io/en/latest/build.html")
try:
    import featexp
except ImportError:
    print("Please install featexp by pip.")
import matplotlib.pyplot as plt

import time
from datetime import datetime
now = datetime.now().strftime('%Y%m%d%H%M%S')
print("Now is {}".format(now))

import warnings
warnings.filterwarnings(action ='ignore')
from hyperopt import hp, fmin, tpe
from sklearn.metrics import accuracy_score
from functools import partial
from sklearn.ensemble import RandomForestClassifier
# import subprocess
# print(subprocess.check_output(['conda','env', 'list']))

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from features import *



train_df = pd.read_csv('D:\MyProjects\CoursML\\assignment2\\all\\train.csv')
test_df = pd.read_csv('D:\MyProjects\CoursML\\assignment2\\all\\test.csv')

# Check for missing data & list them 
nas = pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('Nan in the data sets')
print(nas[nas.sum(axis=1) > 0])


#%%
# Missing values

train_df['Age']=train_df['Age'].fillna(train_df.loc[train_df['Age'].notnull()]['Age'].mean())
train_df['Cabin'] = train_df['Cabin'].fillna(train_df['Cabin'].value_counts().index[0])
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].value_counts().index[0])

test_df['Age']=test_df['Age'].fillna(test_df.loc[train_df['Age'].notnull()]['Age'].mean())
test_df['Cabin'] = test_df['Cabin'].fillna(test_df['Cabin'].value_counts().index[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].value_counts().index[0])
test_df['Fare']=test_df['Fare'].fillna(test_df.loc[train_df['Fare'].notnull()]['Fare'].mean())

# Scale numbers
train_df['Fare'] = (train_df['Fare']-train_df['Fare'].min())/(train_df['Fare'].max()-train_df['Fare'].min())
test_df['Fare'] = (test_df['Fare']-test_df['Fare'].min())/(test_df['Fare'].max()-test_df['Fare'].min())
# Assure no missing values
nas = pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
if nas[nas.sum(axis=1) > 0].empty:
    print("No missing values.")
else:        
    print('Nan in the data sets')
    print(nas[nas.sum(axis=1) > 0])

features = {'Sex': sexFeature(train_df), 
            'EmbarkCoded': embarkedFeature(train_df), 
            "FamilySize": familysizeFeature(train_df), 
            "FamilySurvival": familysurvivalFeature(train_df),
            "Pclass": pclassFeature(train_df)}


featureGen = featureGenerator(train_df, test_df, features = features)
X_test, X_train = featureGen.fit()
y_train = train_df["Survived"]

test_df["Survived"] = RandomForestClassifier(criterion='gini', max_depth=5, n_estimators=110, oob_score= True, random_state= 0).fit(X_train, y_train).predict(X_test) #0.80

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_df['Survived']
    })

file = 'titanic_preditedby{}_{}.csv'.format('RF', now)
submission.to_csv(file, index=False)
print('Exported '+file)