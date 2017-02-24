import numpy as np
from myprocess import get_data

path = './../../machine_learning_examples/ann_logistic_extra'
csv_name = 'ecommerce_data.csv'

#%load_ext autoreload
#autoreload

X, Y = get_data(path, csv_name)

M = 5
D = X.shape[1]
K = len(set(Y))
N = X.shape[0]

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)

W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def softmax(A):
	  expA = np.exp(A)
	  return expA/np.sum(expA, axis=1, keepdims=True) 

def forward(X, W1, b1, W2, b2):
	  Z = np.tanh(X.dot(W1) + b1)
	  Y = softmax(Z.dot(W2) + b2)
	  return Y   

def classification_rate(Y, P):
	  return np.mean(Y == P)


P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis = 1)

print classification_rate(Y, P)
