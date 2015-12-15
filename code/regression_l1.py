from sklearn import linear_model as splm
import numpy as np
from sklearn import preprocessing as pp
import scipy as sc
import matplotlib.pyplot as plt


lag = 1

y = pp.normalize(np.load("../data/filtered_data.npy"))
yvar = np.var(y, axis=1)
print np.max(yvar), np.min(yvar)
varmask = np.where(yvar > .0000001)[0]
print np.max(yvar[varmask]), np.min(yvar[varmask])
y = y[varmask].T

x = np.load("../description_pp/design_matrix_1.npy")
x = x[:len(x)-1]

xtrain, ytrain = x[:3000], y[:3000]
xtest, ytest = x[3000:], y[3000:]


def nonzero(x, y):
	ind = [i for i in range(len(y)) if y[i].any()]
	return x[ind], y[ind]

xtrain, ytrain = nonzero(xtrain, ytrain)
xtest, ytest = nonzero(xtest, ytest)

class lassoreg:
	def __init__(self):
		return

	def train(self, x, y):
		self.models = []
		self.coef = []
		# self.t = []
		# self.p = []
		for i in range(y.shape[1]):
			clf = splm.Lasso(alpha=.000000005)
			clf.fit(x, y[:,i])

			# n, k = x.shape
			# yHat = np.matrix(clf.predict(x)).T
			# xm = np.hstack((np.ones((n,1)),np.matrix(x)))
			# ym = np.matrix(y[:,i]).T
			# df = float(n-k-1)

			# # Sample variance.     
			# sse = np.sum(np.square(yHat - y[:,i]),axis=0)
			# sampleVariance = sse/df

			# # Sample variance for x.
			# sampleVarianceX = np.dot(x.T,x)
			# print sampleVarianceX.shape

			# # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
			# self.covarianceMatrix = sc.linalg.sqrtm(sampleVariance[0,0]*np.linalg.pinv(sampleVarianceX))

			# # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
			# se = covarianceMatrix.diagonal()[1:]

			# # T statistic for each beta.
			# t = np.zeros(len(se))
			# for i in xrange(len(se)):
			# 	t[i] = clf.coef_[0,i]/se[i]

			# # P-value for each beta. This is a two sided t-test, since the betas can be 
			# # positive or negative.
			# p = 1 - t.cdf(abs(t),df)

			self.models.append(clf)
			self.coef.append(clf.coef_)
			# self.t.append(t)
			# self.p.append(p)
			if i%500 == 0: print i, clf.coef_, clf.coef_.any()
		# self.coef = [clf.coef_ for clf in self.models]

	def predict(self, x):
		preds = self.models[0].predict(x)
		for i in range(1, len(self.models)):
			preds = np.vstack((preds, self.models[i].predict(x)))
		return preds.T

	def accuracy(self, pred, y):
		total = 0
		for i in range(len(pred)):
			pi, yi = pred[i], y[i]
			total += np.sum((yi-pi)**2)
		return total

# def plot(y, p):
def plot(pred, y):
	times = range(0, 2*len(pred), 2)
	plt.plot(times, pred, 'r-', times, y, 'b-')
	# plt.plot(y)
	plt.show()

l = lassoreg()
l.train(xtrain, ytrain)
with open("../data/l1coef2.npy", "w") as f:
	np.save(f, l.coef)
pred = l.predict(xtest)
with open("../data/l1preds2.npy", "w") as f:
	np.save(f, pred)
print pred
print pred.shape
acc = l.accuracy(pred, ytest)
print acc



