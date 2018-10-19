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
    
	 
	grad = [theta_grad(theta, X, j, y) for j in range(X.shape()[1])]
    
    
    # =============================================================
	
	return grad

def theta_grad(theta, X, j, y):
	return sum([(product_theta_x(theta, X, i)-y[i])*X[i, j] for i in range(X.shape()[0])])