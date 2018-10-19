import math
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from my_LDA import my_LDA
from predict import predict
from sklearn import preprocessing

# Load data (Wine dataset)
my_data = np.genfromtxt('D:\MyProjects\CoursML\Lab3\MA2823-ML_Lab3\LDA\wine_data.csv', delimiter=',')
np.random.shuffle(my_data) # shuffle datataset

trainingData = my_data[:100,1:] # training data
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(trainingData)
trainingData = minmax_scale.transform(trainingData)
trainingLabels = my_data[:100,0] # class labels of training data

testData = my_data[101:,1:] # training data
testData = minmax_scale.transform(testData)
testLabels = my_data[101:,0] # class labels of training data


# Training LDA classifier
W, projected_centroid, X_lda = my_LDA(trainingData, trainingLabels)

# Perform predictions for the test data
predictedLabels = predict(testData, projected_centroid, W)



# Compute accuracy
counter = 0
for i in range(predictedLabels.size):
    if predictedLabels[i] == testLabels[i]:
        counter += 1
print('Accuracy of LDA: %f' % (counter / float(predictedLabels.size) * 100.0))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(solver = 'eigen')
clf.fit(trainingData, trainingLabels)
test = clf.predict(testData)
counter = 0
for i in range(test.size):
    if test[i] == testLabels[i]:
        counter += 1
print('Accuracy of LDA: %f' % (counter / float(test.size) * 100.0))

clf = LinearDiscriminantAnalysis(solver = 'svd')
clf.fit(trainingData, trainingLabels)
test = clf.predict(testData)
counter = 0
for i in range(test.size):
    if test[i] == testLabels[i]:
        counter += 1
print('Accuracy of LDA: %f' % (counter / float(test.size) * 100.0))