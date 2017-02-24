import numpy as np

a = np.random.randn(5)
expA = enp.xp(a)
sftmax = axpA/axpA.sum()
print sftmax
print sftmax.sum()

a = np.random.randn(100, 5)
sftmax = np.exp(a)/np.exp(a).sum(axis=1, keepdims=True)
print np.exp(a).sum(axis=1, keepdims=True).shape
print sftmax
print sftmax.sum(axis=1)

