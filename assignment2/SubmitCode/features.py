'''
File: /Users/lichenle/Desktop/ML A2/main.py
Project: /Users/lichenle/Desktop/ML A2
Created Date: Sunday November 25th 2018
Author: Chenle Li
-----
Last Modified: 2018-12-02 03:20:12
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
    """Class for generating features over dataset with dictionnaies 
    {feature name: a child class of featureBase() which generate correpondent feature}
    
    """

    def __init__(self, 
                train_df, 
                test_df,  
                features = None):
        """Initilization
        
        Args:
            train_df (pd.DataFrame): Train DataSet
            test_df (pd.DataFrame): Test Dataset
            features (dict, optional): Defaults to None. Dictionary of {feature names: feature()}
        """

        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.features = features
        self.add(features)
        # self.data_df = self.train_df.copy().append(test_df)

    def add(self, feature_dict):
        """add new features with feature names and feature class
        
        Args:
            feature_dict (dict): Dictionary of {feature names: feature()} to be added
        """

        for feature in feature_dict:
            self.features[feature] =  feature_dict[feature]
            self.train_df[feature] = feature_dict[feature].compute(self.train_df.append(self.test_df))[:self.train_df.shape[0]]

    def fit(self):
        """Generate features separately over the train dataset and test dataset.
        
        Returns:
        X_test, X_train: test and train dataset with features generated
        """

        for feature in self.features:
            self.test_df[feature] = self.features[feature].compute(self.train_df.append(self.test_df))[self.train_df.shape[0]:]
        features = list(self.features)
        std_scaler = sklearn.preprocessing.StandardScaler()
        self.train_df[features] = std_scaler.fit_transform(self.train_df[features])
        self.test_df[features] = std_scaler.transform(self.test_df[features])
        X_test = self.test_df[features].copy()
        X_train = self.train_df[features].copy()
        return X_test, X_train

class featureBase():
    """Parent class for generation one feature
    
    Returns:
        [type]: [description]
    """


    def __init__(self, 
                df,
                name):
        self.df = df.copy()
        self.name = name

    # @staticmethod
    def compute(self, dataframe):
        """Do the feature generation job, to be overwritten in sub classes
        
        Args:
            dataframe (pd.DataFrame): the dataset to be generated over
        
        Returns:
            (Series): feature column
        """

        # return a Series type of features
        pass
    
    def runBaseline(self):
        self.df[self.name] = self.compute(self.df)
        baseline(self.name, self.df, Target="Survived")
        return 0

class sexFeature(featureBase):
    """Map sex to numbers [man, woman] -> [0, 1]
    
    Args:
        featureBase (class): parent class
    
    Returns:
        Sex: 0, 1 representation of Sex
    """

    def __init__(self, df):
        super().__init__(df, 'Sex')
        
    def compute(self, dataframe):

        dataframe[self.name].replace(['male','female'],[0,1],inplace=True)
        Sex = dataframe[self.name]
        return Sex


class embarkedFeature(featureBase):
    """Map [S, C, Q] -> [0, 1, 2]
    
    Args:
        featureBase (class): parent class
    
    Returns:
        Embarked (Series): numerical representation of departure cities
    """

    
    def __init__(self, df):
        super().__init__(df, 'Embarked')
    
    def compute(self, dataframe):
        dataframe[self.name].replace(['S', 'C', 'Q'], [0, 1, 2],inplace=True)
        Embarked = dataframe[self.name]
        return Embarked

class familysizeFeature(featureBase):
    """Generate family size by SisSp+Parch (somme of number of children/parents/siblings.)
    
    Args:
        featureBase (class): parent class
    
    Returns:
        familysize (Series): number of family members on board
    """


    def __init__(self, df):   
        super().__init__(df, "FamilySize")

    def compute(self, dataframe):
        familysize = dataframe["SibSp"]+dataframe["Parch"]
        return familysize

class familysurvivalFeature(featureBase):
    """Generate the survival probabilities of family members based on info of survival
    in train dataset.

    The exact family members are determined by LastName and Fare.
    
    Args:
        featureBase (class): parent class
    
    Returns:
        family_survival (Series): family survival rate.
    """


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
        family_survival = data_df["Family_Survival"]
        return family_survival


class pclassFeature(featureBase):
    """Get the raw pclass feature 
    
    Args:
        featureBase (class): parent class
    
    Returns:
       pclass: the raw information of class
    """ 


    def __init__(self, df):
        super().__init__(df, "Pclass")

    def compute(self, dataframe):
        pclass = dataframe["Pclass"]
        return pclass

class titleFeature(featureBase):
    """Get social status based on title in the name
    
    Args:
        featureBase (class): parent class
    
    Returns:
       title: the title categories based on risk of social status [highrisk, mediumrisk, lowrisk] -> [2, 1, 0]
    """ 


    def __init__(self, df):
        super().__init__(df, "Title")
        self.map = {'Mr': 2, 
                    'Don': 2, 
                    'Rev': 2, 
                    'Capt': 2, 
                    'Jonkheer': 2, 
                    'Master': 1, 
                    'Dr': 1, 
                    'Major': 1, 
                    'Col': 1, 
                    'Mrs': 0, 
                    'Miss': 0, 
                    'Mme': 0, 
                    'Ms': 0, 
                    'Lady': 0, 
                    'Sir': 0, 
                    'Mlle': 0, 
                    'the Countess': 0,
                    "Dona": 1
                } 
    
    def compute(self, dataframe):
        dataframe['Title'] = dataframe['Name']
        get_title = lambda x: re.findall(r",\s(.+?)\.", x)[0]
        dataframe['Title'] = dataframe['Title'].apply(get_title)
        dataframe['Title'] = dataframe['Title'].map(self.map)
        title = dataframe['Title']
        return title

