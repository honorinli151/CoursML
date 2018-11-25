'''
File: /Users/lichenle/Desktop/ML A2/main.py
Project: /Users/lichenle/Desktop/ML A2
Created Date: Sunday November 25th 2018
Author: Chenle Li
-----
Last Modified: 2018-11-25 11:02:00
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
'''

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
from utils import *
from models import *

# %mkdir templates

# %%file templates/myhtml.tpl
# {% extends "html.tpl" %}
# {% models
# <h1>{{ table_title|default("My Table") }}</h1>
# {{ super() }}
# {% endblock table %}

class featureGenerator():
    
    def __init__(self, 
                train_df:pd.DataFrame, 
                test_df:pd.DataFrame,  
                features: dict = None):

        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.features = features
        self.add(features)
        # self.data_df = self.train_df.copy().append(test_df)

    def add(self, feature_dict):
        for feature in feature_dict:
            self.features[feature] =  feature_dict[feature]
            self.train_df[feature] = feature_dict[feature].compute(self.train_df.append(self.test_df))[:self.train_df.shape[0]]

    def fit(self):
        for feature in self.features:
            self.test_df[feature] = self.features[feature].compute(self.train_df.append(self.test_df))[self.train_df.shape[0]:]
        features = list(self.features)
        std_scaler = sklearn.preprocessing.StandardScaler()
        self.train_df[features] = std_scaler.fit_transform(self.train_df[features])
        self.test_df[features] = std_scaler.transform(self.test_df[features])
        return self.test_df[features].copy(), self.train_df[features].copy()

class featureBase():

    def __init__(self, 
                df,
                name):
        self.df = df.copy()
        self.name = name

    # @staticmethod
    def compute(self, dataframe):
        # return a Series type of features
        pass
    
    def runBaseline(self):
        self.df[self.name] = self.compute(self.df)
        baseline(self.name, self.df, Target="Survived")

class sexFeature(featureBase):

    def __init__(self, df):
        super().__init__(df, 'Sex')
        
    def compute(self, dataframe):
        dataframe[self.name].replace(['male','female'],[0,1],inplace=True)
        return dataframe[self.name]


class embarkedFeature(featureBase):
    
    def __init__(self, df):
        super().__init__(df, 'Embarked')
    
    def compute(self, dataframe):
        dataframe[self.name].replace(['S', 'C', 'Q'], [0, 1, 2],inplace=True)
        return dataframe[self.name]

class familysizeFeature(featureBase):

    def __init__(self, df):   
        super().__init__(df, "FamilySize")

    def compute(self, dataframe):
        familysize = dataframe["SibSp"]+dataframe["Parch"]
        return familysize

class familysurvivalFeature(featureBase):

    def __init__(self, df):
        super().__init__(df, "FamilySurvival")

    def compute(self, dataframe):
        data_df = dataframe

        data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
        data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

        DEFAULT_SURVIVAL_VALUE = 0.5
        data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

        for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                                'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
            
            if (len(grp_df) != 1):
                # A Family group is found.
                for ind, row in grp_df.iterrows():
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if (smax == 1.0):
                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif (smin==0.0):
                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

        print("Number of passengers with family survival information:", 
            data_df.loc[data_df['Family_Survival']!=0.5].shape[0])

        for _, grp_df in data_df.groupby('Ticket'):
            if (len(grp_df) != 1):
                for ind, row in grp_df.iterrows():
                    if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                        smax = grp_df.drop(ind)['Survived'].max()
                        smin = grp_df.drop(ind)['Survived'].min()
                        passID = row['PassengerId']
                        if (smax == 1.0):
                            data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                        elif (smin==0.0):
                            data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                                
        print("Number of passenger with family/group survival information: " +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))
        return data_df["Family_Survival"]


class pclassFeature(featureBase):

    def __init__(self, df):
        super().__init__(df, "Pclass")

    def compute(self, dataframe):
        pclass = dataframe["Pclass"]
        return pclass