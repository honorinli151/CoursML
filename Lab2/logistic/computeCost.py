from numpy import *
from sigmoid import sigmoid

def computeCost(theta, X, y): 
	# Computes the cost using theta as the parameter 
	# for logistic regression. 
    
	m = X.shape[0] # number of training examples
	
	J = 0
	
	# ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta (see the assignment 
    #				for more details).

    J = -(1/m)*sum([y[i]*log(product_theta_x(theta, X, i))+\
		(1-y[i])*log(1-product_theta_x(theta, X, i)) for i in range(len(y))])

    
    
    
    # =============================================================
	
	return J

def product_theta_x(theta, X, i):
	return 1/(1+exp(theta[i, :].dot(X)))