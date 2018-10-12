'''
File: /Users/lichenle/Desktop/MyProject/CoursML/Lab1/Code/MDS/my_mds.py
Project: /Users/lichenle/Desktop/MyProject/CoursML/Lab1/Code/MDS
Created Date: Thursday September 27th 2018
Author: Chenle Li
-----
Last Modified: 2018-10-12 12:39:40
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
'''

from numpy import *
import matplotlib.pyplot as plt
import scipy.linalg

# Load the distance matrix
D = loadtxt('/Users/lichenle/Desktop/MyProject/CoursML/Lab1/Code/MDS/distanceMatrix.csv', delimiter=',')
cities = ['Atl','Chi','Den','Hou','LA','Mia','NYC','SF','Sea','WDC']
nCities = D.shape[0] # Get the size of the matrix
print(nCities)

k = 2 # e.g. we want to keep 2 dimensions


#================= ADD YOUR CODE HERE ====================================
## TODO: Implement MDS
## Add your code here
# Instructions: Use MDS to reduce the dimensionality of the data
#				while preserving the distances between all pairs
#				of points. Use the steps given in the description 
#				of the assignment. Initially, calculate matrix J
# 				and subsequently, matrix B. Perform SVD decomposition
#				of B. Calculate new representation. Save the new 
#				representation in variable X


D = pow(D, 2)
J = identity(10)-ones((10, 10))/10
B = -0.5*J.dot(D.dot(J))
try:
    U, S, V = linalg.svd(B, full_matrices=True, compute_uv=True)
except LinAlgError:
    print("SVD computation does not converge.")
except:
    print('Check linalg.svd function')

sigma = diag(S)
def disntance(k):
    s = sigma[:k, :k]
    u = U[:, :k]
    v = V[:k, :]
    return u.dot(scipy.linalg.fractional_matrix_power(s, 0.5))

X = disntance(k)


#=================================================================


# Plot distances in two dimensions
plt.figure(1)

# Plot cities in 2D space
plt.subplot(121)
plt.plot(-X[:,0],-X[:,1],'o')
for i in range(len(cities)):
     plt.text(-X[i,0], -X[i,1]+1.5, cities[i], color='k', ha='center', va='center')

# Plot also a US map
plt.subplot(122)
im = plt.imread("/Users/lichenle/Desktop/MyProject/CoursML/Lab1/Code/MDS/usamap.png")
implot = plt.imshow(im,aspect='auto')
plt.axis('off')
plt.show()


