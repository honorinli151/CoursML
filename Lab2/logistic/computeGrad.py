from numpy import *
from sigmoid import sigmoid

def computeGrad(theta, X, y):
	# Computes the gradient of the cost with respect to
	# the parameters.
	
	m = X.shape[0] # number of training examples
	
	grad = zeros(size(theta)) # initialize gradient
	
	# ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.
    
	log_theta_x =1/(1+exp(theta.dot(X)))
	grad = -(1/m)*sum([])
    
    
    
    
    
    
    
    # =============================================================
	
	return grad
