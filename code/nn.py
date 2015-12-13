import numpy as np
import numpy.random as npr
import random as rand
import csv
import scipy.io as scio
import scipy.special as scsp
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt
import time
import pybrain.datasets as pbds
import pybrain.tools.shortcuts as pbts
import pybrain.supervised.trainers as pbsvt

# from sklearn.neural_network import MLPClassifier

# load words

words = np.array(np.load("../description_pp/word_list.p"))
wordlist = words.tolist()
# print len(words)
# print wordlist
print wordlist.index(u'angry.a.01')
print wordlist.index(u'girl.n.01')
print wordlist.index(u'kiss.n.01')


# load data

x = pp.normalize(np.transpose(np.load("../data/masked_data_17k.npy")))
# xvar = np.var(x, axis=1)
# varmask = np.where(xvar > .00012)
# print np.max(xvar[varmask]), np.min(xvar[varmask])
# xtrain = xtrain[varmask].T
# print np.max(xtrain), np.min(xtrain)
# print 
# print ve
# print np.max(ve), np.min(ve), ve.shape
# xtest = (pp.normalize(np.load("../data/BOLD_val_masked34589.npy"))[varmask]).T

lag = 1
xtrain = x[lag:lag+3000]
xtest = x[3000:]

def nonzero(x, y):
	ind = [i for i in range(len(y)) if y[i].any()]
	return x[ind], y[ind]

y = np.load("../description_pp/design_matrix_1.npy")
ytrain = y[:3000]
ytest = y[3000:3000+xtest.shape[0]]

xtrain, ytrain = nonzero(xtrain, ytrain)
xtest, ytest = nonzero(xtest, ytest)

sums = np.sum(ytrain, axis = 0).tolist()
mostcommonwords = [j[0] for j in sorted(enumerate(sums), key=lambda i: i[1], reverse=True)[:10]]

# print xtrain
# print xtest
# print y
# print y[0].tolist()
print xtrain.shape, xtest.shape, ytrain.shape, ytest.shape

class NeuralNetworkNaive():
	def __init__(self, inputSize, hiddenSize, outputSize, mode="ce"):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.hiddenSize = hiddenSize

		self.w1 = npr.normal(0, .01, (inputSize, hiddenSize))
		self.b1 = np.ones(hiddenSize)

		self.w2 = npr.normal(0, .01, (hiddenSize, outputSize))
		self.b2 = np.ones(outputSize)

		print self.w1.shape, self.w2.shape

		self.errors = []

		self.endlearn = None
		
		self.mode = mode

	def forward(self, X):
		# print X.shape, self.w1.shape

		self.z2 = np.dot(X, self.w1) + self.b1
		self.a2 = self.tanh(self.z2)
		
		self.z3 = np.dot(self.a2, self.w2) + self.b2
		self.a3 = self.sigmoid(self.z3)

		# print self.z3
		# print self.a3
		return self.a3

	def backward(self, x, y, a):
		if self.mode == "mse":
			self.d_mse(x, y)
		else:
			self.d_ce(x, y)

		self.w1 -= a*self.djdw1
		self.w2 -= a*self.djdw2
		self.b1 -= a*np.reshape(self.d2, (self.d2.shape[1],))
		self.b2 -= a*np.reshape(self.d3, (self.d3.shape[1],))
		# print self.w1
		# print self.w2
		# print "--------------"

	def train(self, x, y, threshold=.9, epoch=1000):
		self.epoch = epoch
		i = 1
		a = [20, .01]
		# cutoff = .9*len(x)
		# tx, ty = x[:cutoff], y[:cutoff]
		# vx, vy = x[cutoff:], y[cutoff:]
		n = len(x)
		while True:
			x, y = dual_shuffle_array(x, y)
			for e in range(0, len(x)):
				# print "\n=============================================="
				# print "CURITER: "+str(i)#+" "+str(np.mean(self.mse(x, y)))
				# print "x: "+str(x[e])
				# print "y: "+str(y[e])
				# ind = rand.randint()

				self.backward(np.array([x[e]]), np.array([y[e]]), self.learn(i, n, a[1]))
				if i%epoch == 0:
				# print "\n=============================================="
				# print "CURITER: "+str(i)#+" "+str(np.mean(self.mse(x, y)))
					a = self.accuracy(y, self.predict(x))
					self.errors.append(a[1])
					print a, self.learn(i, n, a[1])
					if a[1] > threshold and i > epoch*30:
						self.endlearn = self.learn(i, n, a[1])
						return a[1]
				i += 1

	def learn(self, i, n, a):
		i = ((i/n)/float(4)) + .1
		return float(1) / (10 * i * (a+.1))

	def predict(self, x):
		return np.array([around(self.forward(np.reshape(xi,(1,xi.shape[0]))), .000000001)[0] for xi in x])

	######## ACTIVATION FUNCTIONS G ########
	def sigmoid(self, z):
		def s(x):
			if x >= 0:
				z = np.exp(-x)
				return 1 / (1 + z)
			else:
				z = np.exp(x)
				return z / (1 + z)
		return np.vectorize(s)(z)

	def d_sigmoid(self, z):
		s = self.sigmoid(z)
		return s * (1-s)

	def tanh(self, z):
		return np.tanh(z)

	def d_tanh(self, z):
		return 1 - self.tanh(z)**2

	######### COST FNS ############

	def accuracy(self, y, h):
		score = 0
		for i in range(len(y)):
			# print str(i)+": "+str(y[i])
			if (y[i] == h[i]).all():
				score += 1
		return score, float(score)/len(y)

	def d_mse(self, x, y):
		self.forward(x)

		self.d3 = np.multiply(-(y-self.a3), self.d_sigmoid(self.z3))
		# print "d3: "+str((y-self.a3).shape)+" * "+str(self.d_sigmoid(self.z3).shape)
		self.djdw2 = np.dot(np.transpose(self.a2),self.d3)
		# print "djdw2: "+str(np.transpose(self.a2).shape)+" "+str(d3.shape)

		# print "d2: "+str(d3.shape)+" "+str(np.transpose(self.w2).shape)+\
			# " * "+str(self.d_sigmoid(self.z2).shape)
		self.d2 = np.dot(self.d3, np.transpose(self.w2)) * self.d_tanh(self.z2)
		# print "djdw1: "+str(np.transpose(np.hstack((x, np.ones((x.shape[0], 1), dtype=np.int)))).shape)+" "+str(d2.shape)
		self.djdw1 = np.dot(np.transpose(x), self.d2)

		return self.djdw1, self.djdw2

	def d_ce(self, x, y):
		self.forward(x)

		self.d3 = -(y-self.a3)
		self.djdw2 = np.dot(np.transpose(self.a2), self.d3)

		self.d2 = np.dot(self.d3, np.transpose(self.w2) * self.d_tanh(self.z2))
		self.djdw1 = np.dot(np.transpose(x), self.d2)

	def plot(self, description):
		plt.plot(range(self.epoch, self.epoch*len(self.errors)+1, self.epoch), self.errors)
		if self.mode == "mse":
			plt.title("Mean Squared Error")
		else:
			plt.title("Cross Entropy Error")
		plt.xlabel("Iteration #")
		plt.ylabel("Accuracy")
		if self.mode == "mse":
			plt.savefig("mse6.jpg")
		else:
			plt.savefig("ce"+description+".jpg")

def around(x, b):
	return np.where(x > b, 1, 0)

def dual_shuffle_array(lst, lst2):
	assert len(lst)==len(lst2)
	newlist = lst[:].tolist()
	newlist2 = lst2[:].tolist()
	l = len(lst)
	for i in range(0, l):
		# print len(lst)-i-1
		ind = npr.randint(0,len(lst)-i)
		newlist.append(newlist.pop(ind))
		newlist2.append(newlist2.pop(ind))
	# print np.array(newlist).shape, np.array(newlist2).shape
	return np.array(newlist), np.array(newlist2)

def nnwords(indices, xtrain, xtest, ytrain, ytest, wordlist, threshold, e):
	if indices:
		ytrain = ytrain[:, indices]
		ytest = ytest[:, indices]

		xtrain, ytrain = nonzero(xtrain, ytrain)
		xtest, ytest = nonzero(xtest, ytest)

	nn = NeuralNetworkNaive(xtrain.shape[1], 5000, ytrain.shape[1])
	nn.train(xtrain, ytrain, threshold, e)
	nn.plot("allwords")

	pred = nn.predict(xtest)
	with open("../nnpreds/nnpreds_"+str("mostcommon")+".npy", "w") as f:
		np.save(f, pred)

	acc = nn.accuracy(ytest, pred)
	print "FINAL ACC "+str("mostcommon")+": "+str(acc)

def nnreal(indices, x, y, hidden=5000):
	if indices:
		x = x[lag:]
		y = y[:len(x)]
		x, y = nonzero(x, y)

	numInputFeatures, numOutputFeatures = x.shape[1], y.shape[1]
	ds = pbds.SupervisedDataset(numInputFeatures, numOutputFeatures)
	ds.setField('input', x)
	ds.setField('target', y)
	dstrain, dstest = ds.splitWithProportion(.93)

	
	nn = pbts.buildNetwork(numInputFeatures, hidden, numOutputFeatures, bias=True)
	trainer = pbsvt.BackpropTrainer(nn, dstrain)
	errors = trainer.trainUntilConvergence()

	for i in dstest


nnreal(mostcommonwords, x, y)





# words = [u'angry.a.01', u'girl.n.01', u'kiss.n.01', u'jenny.n.01', u'christmas.n.01']
# words = ytrain[:, words]
nnwords(mostcommonwords, xtrain, xtest, ytrain, ytest, wordlist, .2, 10)



# (37, 0.04596273291925466) 6.85106382979
# (169, 0.20993788819875778) 3.22645290581
# (0, 0.0) 10.0
# (5, 0.006211180124223602) 9.41520467836
# (0, 0.0) 10.0
# (0, 0.0) 10.0
# (89, 0.11055900621118013) 4.74926253687
# (43, 0.05341614906832298) 6.51821862348
# (0, 0.0) 10.0
# (169, 0.20993788819875778) 3.22645290581
# (169, 0.20993788819875778) 3.22645290581
# FINAL ACC [u'rubbish.n.01' u'teammate.n.01' u'abruptly.r.01' ..., u'tray.n.01'
#  u'wheelchair.n.01' u'bench.n.01']: (169, 0.20993788819875778)

pred = np.load("../nnpreds/nnpred_mostcommon.npy").astype(np.int)
print np.where(pred != 0)
# print pred.tolist()
print pred.shape

# words = np.array(np.load("../description_pp/word_list.p"))
# wordlist = words.tolist()
# print len(words)
# print wordlist
# print wordlist.index(u'angry.a.01')
# print wordlist.index(u'girl.n.01')
# print wordlist.index(u'kiss.n.01')












