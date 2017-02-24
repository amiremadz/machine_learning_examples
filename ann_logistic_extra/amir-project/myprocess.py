import numpy as np
import pandas as pd
from os.path import join

#path = './../../machine_learning_examples/ann_logistic_extra'
#csv_name = 'ecommerce_data.csv'


def get_data(path, csv_name):
		df = pd.read_csv(join(path, csv_name))
		df = df.as_matrix()
		#print df.head()

		X = df[:, :-1]
		Y = df[:, -1]

		# Normalize numerical columns
		X[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()
		X[:, 2] = (X[:, 2] - X[:, 2].mean())/X[:, 2].std()

		N, D = X.shape
		X2 = np.zeros((N, D+3))
		X2[:, 0:(D-1)] = X[:, 0:(D-1)]

		for n in xrange(N):
			  t = int(X[n, D-1])
			  X2[n, D-1+t] = 1

		Z = np.zeros((N, 4))
		Z[np.arange(N), X[np.arange(N), D-1].astype(np.int32)] = 1

		return X2, Y

def get_binary_data(path, csv_name):
	  X, Y = get_data(path, csv_name)
	  X2 = X[Y <= 1]
	  Y2 = Y[Y <= 1]
	  return X2, Y2
