'''
File: d:\MyProjects\CoursML\Lab1\Code\SVD\my_svd.py
Project: d:\MyProjects\CoursML\Lab1\Code\SVD
Created Date: Thursday September 27th 2018
Author: Chenle Li
-----
Last Modified: 2018-09-27 09:31:57
Modified By: Chenle Li at <chenle.li@student.ecp.fr>
-----
Copyright (c) 2018 Chenle Li
-----
HISTORY:
Date               	  By     	Comments
-------------------	---------	---------------------------------------------------------
'''

"""
Ref: For expication overall: https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
"""

from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Load the "gatlin" image data
X = loadtxt('d:/MyProjects/CoursML/Lab1/Code/SVD/gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the 
#               three factors in variables U, S and V
# Ref: SVD API  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html 

try:
    U, S, V = linalg.svd(X, full_matrices=True, compute_uv=True)
except LinAlgError:
    print("SVD computation does not converge.")
except:
    print('Check linalg.svd function')


#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create five matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values 
#  Ref: API mutiplication matrix: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.dot.html
#       API create a diag matrix: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.diag.html

K = [10, 20, 50, 100, 200]
sigma = diag(S)

def reconstruction(k):
    s = sigma[:k, :k]
    u = U[:, :k]
    v = V[:k, :]
    return u.dot(s.dot(v))

Stock_S = [] 
for k in K:
    Stock_S.append(reconstruction(k))

#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
# Ref: Frobenius error API https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.norm.html

errors = [] 
for x in Stock_S:
    errors.append(linalg.norm(subtract(X, x), ord=None))
print("The errors are: \n "+"K: "+str(K)[1:-1]+'\n'+'E: '+str(errors))

#=========================================================================



# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

X10, X20, X50, X100, X200 = Stock_S

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X10,cmap = cm.Greys_r)
plt.title('Best rank' + str(10) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(X20,cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X50,cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X100,cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X200,cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k

plt.figure(3)
plt.plot(list(range(len(S))), S)
plt.xlabel('Rank')
plt.ylabel('Singular Value')
plt.show()

#=========================================================================

plt.show() 

