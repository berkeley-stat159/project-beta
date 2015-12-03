import numpy as np
import numpy.random as npr
import random as rand
import csv
import scipy.io as scio
import scipy.special as scsp
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt
import time
# from sklearn.neural_network import MLPClassifier


xtrain = pp.normalize(np.load("../data/BOLD_est_masked34589.npy"))
xvar = np.var(xtrain, axis=1)
varmask = np.where(xvar > .00015)
print np.max(xvar[varmask]), np.min(xvar[varmask])
xtrain = xtrain[varmask].T
print np.max(xtrain), np.min(xtrain)
# print 
# print ve
# print np.max(ve), np.min(ve), ve.shape

xtest = (pp.normalize(np.load("../data/BOLD_val_masked34589.npy"))[varmask]).T
y = np.load("../description_pp/design_matrix_1.npy")
ytrain = y[:3000]
ytest = y[3000:3000+xtest.shape[0]]
# print xtrain
# print xtest
# print y
# print y[0].tolist()
print xtrain.shape, xtest.shape, y.shape, ytrain.shape, ytest.shape

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

	def train(self, x, y):
		i = 1
		a = [20, .01]
		cutoff = .9*len(x)
		tx, ty = x[:cutoff], y[:cutoff]
		vx, vy = x[cutoff:], y[cutoff:]
		n = cutoff-1
		while True:
			x, y = dual_shuffle_array(x, y)
			tx, ty = x[:cutoff], y[:cutoff]
			vx, vy = x[cutoff:], y[cutoff:]
			for e in range(0, len(tx)):
				# print "\n=============================================="
				# print "CURITER: "+str(i)#+" "+str(np.mean(self.mse(x, y)))
				# print "x: "+str(x[e])
				# print "y: "+str(y[e])
				# ind = rand.randint()

				self.backward(np.array([tx[e]]), np.array([ty[e]]), self.learn(i, n, a[1]))
				# if i%50 == 0:
				# print "\n=============================================="
				# print "CURITER: "+str(i)#+" "+str(np.mean(self.mse(x, y)))
			a = self.accuracy(vy, self.predict(vx))
			self.errors.append(a[1])
			print a
			if a[1] > .44:
				self.endlearn = self.learn(i, n, a[1])
				return a[1]
				# i += 1

	def learn(self, i, n, a):
		i = (float(i/n)/4) + 1
		return float(1) / (10 * i) / 20

	def predict(self, x):
		return np.array([np.around(self.forward(np.reshape(xi,(1,xi.shape[0]))))[0] for xi in x])

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

	def plot(self):
		plt.plot(range(1000, 1000*len(nn.errors)+1, 1000), nn.errors)
		if self.mode == "mse":
			plt.title("Mean Squared Error")
		else:
			plt.title("Cross Entropy Error")
		plt.xlabel("Iteration #")
		plt.ylabel("Accuracy")
		if self.mode == "mse":
			plt.savefig("mse6.jpg")
		else:
			plt.savefig("ce6.jpg")

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


nn = NeuralNetworkNaive(xtrain.shape[1], 1000, ytrain.shape[1])
nn.train(xtrain, ytrain)
nn.plot()

pred = nn.predict(xtest)
with open("nnpreds.npy", "w") as f:
	np.save(f, pred)

acc = nn.accuracy(ytest, pred)
print "FINAL ACC: "+str(acc)




















