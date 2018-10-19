from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from computeCost import computeCost
from computeGrad import computeGrad
from predict import predict
from sklearn.linear_model import LogisticRegression

# Load the dataset
# The first two columns contains the exam scores and the third column
# contains the label.
data = loadtxt('D:\MyProjects\CoursML\Lab2\logistic\data1.txt', delimiter=',')
 
X = data[:, 0:2]
y = data[:, 2]

print(X)
print(c_[ones(X.shape[0]), X])

# Plot data 
pos = where(y == 1) # instances of class 1
neg = where(y == 0) # instances of class 0
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.show()


#Add intercept term to X
X_new = 0.5*ones((X.shape[0], 3))
X_new[:, 1:3] = X
X = X_new
# X = c_[ones((data.shape[0],1)), X[:,0:2]]
# print(X-X_new)

# Initialize fitting parameters
initial_theta = array([-12, 0.1, 0.1])

# Run minimize() to obtain the optimal theta
Result = op.minimize(fun = computeCost, x0 = initial_theta, args = (X, y), method = None,jac = computeGrad)


theta = Result.x
print(Result)

# clf = LogisticRegression(random_state=0, solver='lbfgs',
#                         multi_class='multinomial').fit(X, y)

# parmas = clf.coef_

# print(parmas)
# theta = c_[clf.intercept_, clf.coef_][0]
# print(theta)
# X = X_new

# pr-1.25532361e+01  9.22224096e-08  1.02895465e-01  1.00512696e-0int('theta', str(theta))

# Plot the decision boundary
plot_x = array([min(X[:, 1]) - 2, max(X[:, 2]) + 2])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + 0.5*theta[0])
plt.plot(plot_x, plot_y)
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
plt.show()

# Compute accuracy on the training set
p = predict(array(theta), X)
counter = 0
for i in range(y.size):
    if p[i] == y[i]:
        counter += 1
print('Train Accuracy: %f' % (counter / float(y.size) * 100.0))
