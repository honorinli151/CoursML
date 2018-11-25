'''
File: f:\ML A2\models.py
Project: f:\ML A2
Created Date: Sunday November 25th 2018
Author: Chenle Li
-----
Last Modified: 2018-11-25 09:59:19
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
'''
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
import pandas as pd


# Machine Learning Algos
MLA = [
        #Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        #Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),

        #GLM
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),

        #Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        #Nearest Neighbor
        neighbors.KNeighborsClassifier(),

        #SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        #Trees    
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        #Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),


        #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        XGBClassifier()    
        ]


def baseline(features:list, train_df:pd.DataFrame, Target="Survived"):
    
    features.append(Target)
    data1 = train_df.copy()[features]
    features.pop()
    Target = [Target]
    

    #split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    #note: this is an alternative to train_test_split
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

    #create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Time','MLA Parameters',]
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #create table to compare MLA predictions
    MLA_predict = data1[Target]

    #index through MLA and save performance to table
    row_index = 0
    for alg in MLA:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg, data1[features].values, data1[Target].values.ravel(), cv  = cv_split)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   


        #save MLA predictions - see section 6 for usage
        alg.fit(data1[features], data1[Target])
        MLA_predict[MLA_name] = alg.predict(data1[features])

        row_index+=1


    #display and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    # display(HTML(MyStyler(MLA_compare).render(table_title="Model Cross Validation Accuracies with feature {}".format('+'.join(features)))))
    return MLA_compare
    

def autoTuning(X, y):
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )

    #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    #removed models w/o attribute 'predict_proba' required for vote classifier and models with a 1.0 correlation to another model
    vote_est = [
        #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
        ('ada', ensemble.AdaBoostClassifier()),
        ('bc', ensemble.BaggingClassifier()),
        ('etc',ensemble.ExtraTreesClassifier()),
        ('gbc', ensemble.GradientBoostingClassifier()),
        ('rfc', ensemble.RandomForestClassifier()),

        #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
        ('gpc', gaussian_process.GaussianProcessClassifier()),
        
        #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        ('lr', linear_model.LogisticRegressionCV()),
        
        #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
        ('bnb', naive_bayes.BernoulliNB()),
        ('gnb', naive_bayes.GaussianNB()),
        
        #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
        ('knn', neighbors.KNeighborsClassifier()),
        
        #SVM: http://scikit-learn.org/stable/modules/svm.html
        ('svc', svm.SVC(probability=True)),
        
        #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    ('xgb', XGBClassifier())

    ]


    #Hard Vote or majority rules
    vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
    vote_hard_cv = model_selection.cross_validate(vote_hard, X, y, cv  = cv_split)
    vote_hard.fit(X, y)

    print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
    print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
    print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
    print('-'*10)


    #Soft Vote or weighted probabilities
    vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
    vote_soft_cv = model_selection.cross_validate(vote_soft, X, y, cv  = cv_split)
    vote_soft.fit(X, y)

    print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
    print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
    print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
    print('-'*10)


    #Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_n_estimator = range(10, 300, 50)
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .75, .1, .15, .25]
    grid_max_depth = [2, 3, 4, 5, 6, 7, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]


    grid_param = [
                [{
                #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
                'n_estimators': grid_n_estimator, #default=50
                'learning_rate': grid_learn, #default=1
                #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
                'random_state': grid_seed
                }],
        
        
                [{
                #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
                'n_estimators': grid_n_estimator, #default=10
                'max_samples': grid_ratio, #default=1.0
                'random_state': grid_seed
                }],

        
                [{
                #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
                'n_estimators': grid_n_estimator, #default=10
                'criterion': grid_criterion, #default=”gini”
                'max_depth': grid_max_depth, #default=None
                'random_state': grid_seed
                }],


                [{
                #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
                #'loss': ['deviance', 'exponential'], #default=’deviance’
                'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
                'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
                #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
                'max_depth': grid_max_depth, #default=3   
                'random_state': grid_seed
                }],

        
                [{
                #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
                'n_estimators': grid_n_estimator, #default=10
                'criterion': grid_criterion, #default=”gini”
                'max_depth': grid_max_depth, #default=None
                'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
                'random_state': grid_seed
                }],
        
                [{    
                #GaussianProcessClassifier
                'max_iter_predict': grid_n_estimator, #default: 100
                'random_state': grid_seed
                }],
            
        
                [{
                #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
                'fit_intercept': grid_bool, #default: True
                #'penalty': ['l1','l2'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
                'random_state': grid_seed
                }],
                
        
                [{
                #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
                'alpha': grid_ratio, #default: 1.0
                }],
        
        
                #GaussianNB - 
                [{}],
        
                [{
                #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
                'n_neighbors': [6,7,8,9,10,11,12,14,16,18,20,22], #default: 5
                'weights': ['uniform', 'distance'], #default = ‘uniform’
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': list(range(1,50,5))
                }],
                
        
                [{
                #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
                #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
                #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [1,2,3,4,5], #default=1.0
                'gamma': grid_ratio, #edfault: auto
                'decision_function_shape': ['ovo', 'ovr'], #default:ovr
                'probability': [True],
                'random_state': grid_seed
                }],

        
                [{
                #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
                'learning_rate': grid_learn, #default: .3
                'max_depth': [1,2,4,6,8,10], #default 2
                'n_estimators': grid_n_estimator, 
                'seed': grid_seed  
                }]   
            ]



    start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
    for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

        #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
        #print(param)
        
        
        start = time.perf_counter()        
        best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc', n_jobs=-1)
        best_search.fit(X, y)
        run = time.perf_counter() - start

        best_param = best_search.best_params_
        print('The best {} parameter for {} is {} with a runtime of {:.2f} seconds.'.format(best_search.best_score_, clf[1].__class__.__name__, best_param, run))
        clf[1].set_params(**best_param) 


    run_total = time.perf_counter() - start_total
    print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

    print('-'*10)

    #%% [markdown]
    # # Submission

    #%%
    #Hard Vote or majority rules w/Tuned Hyperparameters
    grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
    grid_hard_cv = model_selection.cross_validate(grid_hard, X, y, cv  = cv_split)
    grid_hard.fit(X, y)

    print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
    print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
    print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
    print('-'*10)

    #Soft Vote or weighted probabilities w/Tuned Hyperparameters
    grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
    grid_soft_cv = model_selection.cross_validate(grid_soft, X, y, cv  = cv_split)
    grid_soft.fit(X, y)

    print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
    print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
    print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
    print('-'*10)
