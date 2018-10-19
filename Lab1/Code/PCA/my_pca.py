'''
File: /Users/lichenle/Desktop/MyProject/CoursML/Lab1/Code/PCA/my_pca.py
Project: /Users/lichenle/Desktop/MyProject/CoursML/Lab1/Code/PCA
Created Date: Thursday September 27th 2018
Author: Chenle Li
-----

=======
Last Modified: 2018-10-19 04:41:10
>>>>>>> ad8d3804ec9a614e2a090df2e53ea183448b6b10
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
2018-09-01 11:34:11	Chenle Li	Normalization added, clusters appeared.
'''

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from numpy import genfromtxt
import csv
from sklearn.preprocessing import StandardScaler

# Load the data set (wine). Variable data stores the final data (178 x 13)
my_data = genfromtxt('D:\MyProjects\CoursML\Lab1\Code\PCA\wine_data.csv', delimiter=',')
data = my_data[:,1:]
target= my_data[:,0] # Class of each instance (1, 2 or 3)
print(target)
print("Size of the data (rows, #attributes) ", data.shape)


# Draw the data in 3/13 dimensions (Hint: experiment with different combinations of dimensions)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(data[:,3],data[:,1],data[:,2], c=target)
ax1.set_xlabel('1st dimension')
ax1.set_ylabel('2nd dimension')
ax1.set_zlabel('3rd dimension')
ax1.set_title("Vizualization of the dataset (3 out of 13 dimensions)")



#================= ADD YOUR CODE HERE ====================================
# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
# Rows of A correspond to observations (i.e., wines), columns to variables.
## TODO: Implement PCA
# Instructions: Perform PCA on the data matrix A to reduce its
#				dimensionality to 2 and 3. Save the projected
#				data in variables newData2 and newData3 respectively
#
# Note: To compute the eigenvalues and eigenvectors of a matrix
#		use the function eigval,eigvec = linalg.eig(M)
# Ref: Covariance API: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.cov.html#numpy.cov
# Ref: Eigenvalues API: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html

M = mean(data.T, axis=1)
C = subtract(data, M)

sc = StandardScaler()
C = sc.fit_transform(C)
print("Size of the C (rows, #attributes) ", C.shape)
W = cov(C, rowvar=False)
print("Size of the W (rows, #attributes) ", W.shape)
w, v = linalg.eig(W)
print("Size of the v  (rows, #attributes) ", v.shape)
print(w)

idx = argsort(w)
print(w[idx[-1]])

## newData2

U2 = concatenate(([v[:,idx[-1]]], [v[:,idx[-2]]]), axis=0)
U2 = U2.T
print("Size of the U2 (rows, #attributes) ", U2.shape)
newData2 = C.dot(U2)
print("Size of the newData2 (rows, #attributes) ", newData2.shape)

## newData3

U3 = concatenate((U2.T, [v[:, idx[-3]]]), axis=0)
U3 = U3.T
print("Size of the U3 (rows, #attributes) ", U3.shape)
newData3 = C.dot(U3)
print("Size of the newData3 (rows, #attributes) ", newData3.shape)




#=============================================================================


 # Plot the first two principal components 
plt.figure(2)
plt.scatter(newData2[:,0],newData2[:,1], c=target)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title("Projection to the top-2 Principal Components")
plt.draw()



# Plot the first three principal components 
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(newData3[:,0],newData3[:,1], newData3[:,2], c=target)
ax.set_xlabel('1st Principal Component')
ax.set_ylabel('2nd Principal Component')
ax.set_zlabel('3rd Principal Component')
ax.set_title("Projection to the top-3 Principal Components")
plt.show()  
