import numpy as np
import scipy as sp
import scipy.linalg as linalg

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    
    classLabels = np.unique(Y) # different class labels on the dataset
    classNum = len(classLabels)
    datanum, dim = X.shape # dimensions of the dataset
    print(dim)
    totalMean = np.mean(X,0) # total mean of the data

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.
    
    # Calculate every centroid
    mean_vectors = []
    for i in range(3):
        mean_vectors.append(np.mean(X[Y==i+1], axis=0))
        print('Mean Vector class %s: %s\n' %(i+1, mean_vectors[i-1]))
    
    
    # Calculate S_w
    S_W = np.zeros((dim, dim))
    
    for cl,mv in zip(range(1,4), mean_vectors):
        class_sc_mat = np.zeros((dim,dim))                  # scatter matrix for every class
        for row in X[Y == cl]:
            row, mv = row.reshape(dim,1), mv.reshape(dim,1) # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                             # sum class scatter matrices
    print('within-class Scatter Matrix:\n', S_W)

    
    # Calculate S_b
    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((dim, dim))
    for i,mean_vec in enumerate(mean_vectors):  
        n = X[Y==i+1,:].shape[0]
        S_B += n * (mean_vec - overall_mean).T.dot(mean_vec - overall_mean)
    
    print('between-class Scatter Matrix:\n', S_B.shape)

    # Solve the eigenvalues problem
    target = np.linalg.inv(S_W)
    print(target.shape)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    print("Eigen Problem Done")
    idx = np.argsort(eig_vals)
    # print(w[idx[-1]])
    # Calculate W
    W = np.concatenate(([eig_vecs[:,idx[-1]]], [eig_vecs[:,idx[-2]]]), axis=0)
    W = W.T
    print('W\n', W)

    projected_centroid = [mean.T.dot(W) for mean in mean_vectors]
    X_lda = X.dot(W)
    # =============================================================

    return W, projected_centroid, X_lda