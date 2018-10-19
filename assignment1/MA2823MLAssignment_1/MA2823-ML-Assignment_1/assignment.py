'''
File: d:\MyProjects\CoursML\assignment1\MA2823MLAssignment_1\MA2823-ML-Assignment_1\assignment.py
Project: d:\MyProjects\CoursML\assignment1\MA2823MLAssignment_1\MA2823-ML-Assignment_1
Created Date: Friday October 19th 2018
Author: Chenle Li
-----
Last Modified: 2018-10-19 12:43:02
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
'''

from numpy import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
import matplotlib.pyplot as plt
from timeit import default_timer as timer

train = loadtxt('D:\MyProjects\CoursML\\assignment1\MA2823MLAssignment_1\MA2823-ML-Assignment_1\data.csv', delimiter=',')

Y = train[:, 0:1]
X = train[:, 1:train.shape[1]]

test = loadtxt('D:\MyProjects\CoursML\\assignment1\MA2823MLAssignment_1\MA2823-ML-Assignment_1\\test.csv', delimiter=',')

Y_test = test[:, 0:1]
X_test = test[:, 1:test.shape[1]]

## Logistic Regression without Feature Selection

start = timer()

clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X, Y.ravel())
prediction = clf.predict(X_test)
print(prediction)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction, pos_label=1)
print(fpr)
roc_auc = metrics.auc(fpr, tpr)

end = timer()
running_time = end -start

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('LogisticRegression  Classifier ROC')
plt.plot(fpr, tpr, color='blue', lw=2, label='SVM ROC area = %0.2f' % roc_auc)
plt.text(0.65, 0.1, 'running time %0.4f s' % running_time)
plt.legend(loc="lower right")
plt.show()


# Logistic Regression with Feature Selection

class logistic_regression_selection:
    def __init__(self, feature_num, Y, X, Y_test, X_test):
        self.feature_num = feature_num
        self.Y = Y
        self.X = X
        self.Y_test = Y_test
        self.X_test = X_test   
    
    def test(self):
        estimator = LogisticRegression(random_state=0, solver='lbfgs')
        selector = RFE(estimator, self.feature_num, step=1)
        start = timer()
        selector = selector.fit(X, Y.ravel())
        end = timer()
        running_time = end - start
        prediction = selector.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        print("Train for feature_num="+str(self.feature_num)+' done')
        return running_time, roc_auc

feature_nums = [20, 40, 60, 80, 100, 150]

running_times = []
roc_scores = []

for feature_num in feature_nums:
    model = logistic_regression_selection(feature_num, Y, X, Y_test, X_test)
    result = model.test()
    running_times.append(result[0])
    roc_scores.append(result[1])

# plt.figure()
# plt.xlabel('Feature Numbers')
# plt.plot(feature_nums, running_times, label="Training Time")
# plt.plot(feature_nums, roc_scores, label="Roc Score")
# plt.legend(loc='lower right')
# plt.show()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Feature Numbers')
ax1.set_ylabel('Training Time', color=color)
ax1.plot(feature_nums, running_times, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Roc Score', color=color)  # we already handled the x-label with ax1
ax2.plot(feature_nums, roc_scores, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
