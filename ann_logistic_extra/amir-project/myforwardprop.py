import numpy as np
from matplotlib import pyplot as plt

D = 2
N = 500
X1 = np.random.randn(N, D) + np.array([0, -2])
X2 = np.random.randn(N, D) + np.array([2, 2])
X3 = np.random.randn(N, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])
Y = np.array([0]*Nsamples + [1]*Nsamples + [2]*Nsamples)

K = 3 
M = 3
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)


plt.figure()
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha = 0.5)
plt.show()

def sigmoid(A):
	  expA = np.exp(-A)
	  return 1/(1 + expA)

def softmax(A):
		expA = np.exp(A)
		return expA/expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
	  Z = sigmoid(X.dot(W1) + b1)
	  Y = softmax(Z.dot(W2) + b2)
	  return Y

def classification_rate(Y, P):
		total_samples = 0
		total_match = 0
		for i in xrange(len(Y)):
			total_samples += 1
			if Y[i] == P[i]:
				total_match += 1

 		return float(total_match)/total_samples


P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)
assert len(Y) == len(P)

print classification_rate(Y, P)







