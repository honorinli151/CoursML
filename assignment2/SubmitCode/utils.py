'''
File: /Users/lichenle/Desktop/ML A2/utils.py
Project: /Users/lichenle/Desktop/ML A2
Created Date: Sunday November 25th 2018
Author: Chenle Li
-----
Last Modified: 2018-11-29 01:46:47
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
'''
import pandas as pd
from datetime import datetime
import pickle   

def submit(test_prediction, label, features, test_df):

    now = datetime.now().strftime('%Y%m%d%H%M%S')
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_prediction
    })
    submission.to_csv('\titanic_baseline_{}_with_{}at{}.csv'.format(label, '+'.join(features), now), index=False)
    
    print('Exported')

def save_model(classifier):
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = 'finalized_model{}at{}.sav'.format(classifier.__class__.__name__, now)
    pickle.dump(classifier, open(filename, 'wb'))
    print('Model saved to {}'.format(filename))
