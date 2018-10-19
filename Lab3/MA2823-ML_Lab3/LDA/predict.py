import numpy as np
import scipy as sp
import scipy.linalg as linalg


def predict(X, projected_centroid, W):
    
    """Apply the trained LDA classifier on the test data 
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """


    # Project test data onto the LDA space defined by W
    projected_data  = np.dot(X, W)
    # print(projected_data.shape)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in the code to implement the classification
    # part of LDA. Follow the steps given in the assigment.
    labels = []
    for row in projected_data:
        # print(projected_centroid)
        distances = [sp.spatial.distance.euclidean(row, center) for center in projected_centroid]
        label = distances.index(min(distances))
        labels.append(label+1)
    labels = np.array(labels)
    
    
    # =============================================================

    # Return the predicted labels of the test data X
    return labels