
#%%
import numpy as np
import pandas as pd
import sklearn
try:
    import xgboost 
except ImportError:
    print("Please install xgboost, refer to https://xgboost.readthedocs.io/en/latest/build.html")
try:
    import featexp
except ImportError:
    print("Please install featexp by pip.")
import matplotlib.pyplot as plt

# import subprocess
# print(subprocess.check_output(['conda','env', 'list']))


#%%
train_df = pd.read_csv('/Users/lichenle/Desktop/MyProject/CoursML/assignment2/all/train.csv')
test_df = pd.read_csv('/Users/lichenle/Desktop/MyProject/CoursML/assignment2/all/test.csv')


#%%
train_df.info()
test_df.info()
train_df.head()

#%% [markdown]
# # Encoding

#%%
train_df.loc[train_df['Sex']=='male', 'Sex']=1
train_df.loc[train_df['Sex']=='female', 'Sex']=0
test_df.loc[test_df['Sex']=='male', 'Sex']=1
test_df.loc[test_df['Sex']=='female', 'Sex']=0

train_df.loc[train_df['Embarked']=='S', 'Embarked']=0
train_df.loc[train_df['Embarked']=='C', 'Embarked']=1
train_df.loc[train_df['Embarked']=='Q', 'Embarked']=2

test_df.loc[test_df['Embarked']=='S', 'Embarked']=0
test_df.loc[test_df['Embarked']=='C', 'Embarked']=1
test_df.loc[test_df['Embarked']=='Q', 'Embarked']=2

#%% [markdown]
# # EDA

#%%
# Check for missing data & list them 
nas = pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('Nan in the data sets')
print(nas[nas.sum(axis=1) > 0])

#%% [markdown]
# ## Plots with featexp

#%%
nums = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']
for feature in nums:
    featexp.get_univariate_plots(data=train_df.loc[train_df[feature].notnull()], target_col='Survived',
                          features_list=[feature], bins=10)
featexp.get_univariate_plots(data=train_df.loc[train_df['Age'].notnull()], target_col='Survived',
                      features_list=['Age'], bins=10)


#%%
import re
# print(re.findall(r"\s(.+?)\.", 'Braund, Mr. Owen Harris')[0])
get_title = lambda x: re.findall(r",\s(.+?)\.", x)[0]
print(train_df['Name'].apply(get_title).value_counts())
print(test_df['Name'].apply(get_title).value_counts())


#%%
title_df = train_df['Name', 'Survived']
title_

#%% [markdown]
# ## Plots

#%%
print(train_df.loc[(train_df['Age']<14) & (train_df['Sex']==1) & (train_df['Survived']==1)].shape[0] / train_df.loc[(train_df['Age']<14) & (train_df['Sex']==1)].shape[0])
train_df.loc[(train_df['Age']<14) & (train_df['Sex']==0) & (train_df['Survived']==1)].shape[0] / train_df.loc[(train_df['Age']<14) & (train_df['Sex']==0)].shape[0]

#%% [markdown]
# ## Feature Generation

#%%
## Generate tranch of ages
train_df.loc[train_df["Age"]<14, "Age"] = 1
train_df.loc[train_df["Age"]<25, "Age"] = 2
train_df.loc[train_df["Age"]<31, "Age"] = 3
train_df.loc[train_df["Age"]<41, "Age"] = 4
train_df.loc[train_df["Age"]>=41, "Age"] = 4

test_df.loc[test_df["Age"]<14, "Age"] = 1
test_df.loc[test_df["Age"]<25, "Age"] = 2
test_df.loc[test_df["Age"]<31, "Age"] = 3
test_df.loc[test_df["Age"]<41, "Age"] = 4
test_df.loc[test_df["Age"]>=41, "Age"] = 4

#%% [markdown]
# # Data Cleaning

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

#%% [markdown]
# # Tuning parameters

#%%
import warnings
warnings.filterwarnings(action ='ignore',category = DeprecationWarning)
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe
from sklearn.metrics import accuracy_score
from functools import partial
from sklearn.ensemble import RandomForestClassifier

features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Sex']
X = train_df[features]
y = train_df['Survived']
def auto_turing(args):
#     model = XGBClassifier(n_jobs = 4, n_estimators = args['n_estimators'],max_depth=6, objective='binary:hinge')
    model = RandomForestClassifier(n_estimators = args['n_estimators'],max_depth=6)
    model.fit(X,y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(accuracy, args)
    return -accuracy

algo = partial(tpe.suggest, n_startup_jobs=4)
space = {"n_estimators":hp.choice("n_estimators",range(20,100,5))}
print(fmin)
best = fmin(auto_turing, space, algo=algo, max_evals=30)
print(best)

#%% [markdown]
# # Train

#%%

clf = RandomForestClassifier(n_estimators = 45,max_depth=6)
# clf = XGBClassifier(n_jobs = 4, n_estimators = 60,max_depth=6)
clf.fit(X, y)
Y_pred = clf.predict(test_df[features])


#%%
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
print('Exported')


