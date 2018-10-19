from numpy import *
from sigmoid import sigmoid
from computeCost import product_theta_x

def computeGrad(theta, X, y):
	# Computes the gradient of the cost with respect to
	# the parameters.
	
	m = X.shape[0] # number of training examples
	
	grad = zeros(size(theta)) # initialize gradient
	
	# ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.
    
	 
	# grad = [theta_grad(theta, X, j, y) for j in range(X.shape[1])]
	grad = (1/m)*((sigmoid(X.dot(theta))-y).T.dot(X))
	# print(grad.shape)
	# print(theta.shape)
	# print(X.shape)
	# print(y.shape)
    
    # =============================================================
	
	print('grad' + str(grad))
	return(grad.flatten())

def theta_grad(theta, X, j, y):
<<<<<<< HEAD
	return sum([(product_theta_x(theta, X, i)-y[i])*X[i, j] for i in range(X.shape()[0])])
=======
	return (1/X.shape[0])*sum([(product_theta_x(theta, X, i)-y[i])*X[i, j] for i in range(X.shape[0])])
>>>>>>> 70cf9cb9c144d6bc727c6b75843d4c806a4ed92e
