from numpy import *
from math import e
from math import pow
from scipy.special import expit

def sigmoid(z):
	# Computes the sigmoid of z.

	# ====================== YOUR CODE HERE ======================
    # Instructions: Implement the sigmoid function as given in the
    # assignment.
    
    # return 1/(1+pow(e, -z))
    return expit(z)