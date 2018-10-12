from numpy import *
from sigmoid import sigmoid

def predict(theta, X):
	# Predict whether the label is 0 or 1 using learned logistic 
	# regression parameters theta. The threshold is set at 0.5
	
	m = X.shape[0] # number of training examples
	
	c = zeros(m) # predicted classes of training examples
	
	p = zeros(m) # logistic regression outputs of training examples
	
	
	# ====================== YOUR CODE HERE ======================
    # Instructions: Predict the label of each instance of the
    #				training set.
	p = theta.dot(X.T)
    
	c = [1 if P>0.5 else 0 for P in p]

    # =============================================================
	
	return c

