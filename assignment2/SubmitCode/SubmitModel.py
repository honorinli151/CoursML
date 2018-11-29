'''
File: /Users/lichenle/Desktop/MyProject/CoursML/assignment2/SubmitCode/pretrainedModel.py
Project: /Users/lichenle/Desktop/MyProject/CoursML/assignment2/SubmitCode
Created Date: Thursday November 29th 2018
Author: Chenle Li
-----
Last Modified: 2018-11-29 02:14:09
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
'''
import pickle
import pandas as pd
from datetime import datetime
now = datetime.now().strftime('%Y%m%d%H%M%S')
print("Now is {}".format(now))


submitModelPath = "finalized_modelRandomForestClassifierat20181129134604.sav"

features = ['Sex', 
            'EmbarkCoded', 
            "FamilySize", 
            "FamilySurvival",
            "Pclass"]

test_df = pd.read_csv("test_ready.csv")
loaded_model = pickle.load(open(submitModelPath, 'rb'))
test_df["Survived"] = loaded_model.predict(test_df[features])

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_df['Survived']
    })

file = 'titanic_preditedby{}_{}.csv'.format("submitModel", now)
submission.to_csv(file, index=False)
print('Exported '+file)