import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys
import argparse

def	get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('data', help='the path to the .csv file that contains data')
	parser.add_argument('mode', help='choose between learning and prediction')
	options = parser.parse_args()
	return options

def Normalize_data(data):
	mean = np.mean(data, axis=0)
	scale = np.std(data - mean, axis=0)
	return (data - mean) / scale

def get_data(file):
	try:
		data = pd.read_csv(file, header=None)
	except:
		print("This csv doesn't work")
		exit(1)
	Y = np.array([data[1]])
	Y = np.where(Y == 'M', 1, 0)
	X = np.array(data.iloc[:,2:])
	X = Normalize_data(X)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y.T, test_size=0.2, random_state=1)
	X_train = X_train.T
	X_test = X_test.T
	Y_train = Y_train.reshape((1, len(Y_train)))
	Y_test = Y_test.reshape((1, len(Y_test)))
	return X_train, X_test, Y_train, Y_test

def layer_sizes(X, Y):
	n_x = X.shape[0]
	n_h = 3
	n_y = Y.shape[0]
	return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
	np.random.seed(42)
	parameters = {	"W1": np.random.randn(n_h, n_x) * 0.01,
					"b1": np.zeros((n_h, 1)),
					"W2": np.random.randn(n_y, n_h) * 0.01,
					"b2": np.zeros((n_y, 1))}
	return parameters

def softmax(z):
	epsilon = 1e-6
	return (np.exp(z)/np.sum(np.exp(z) + epsilon, axis=0, keepdims=True))

def sigmoid(x):
	s = 1/(1+np.exp(-x))
	return s

def dsigmoid(dA, Z):
	s = 1/(1+np.exp(-Z))
	ds = dA * s * (1 - s)
	return ds

def drelu(dA, Z):
	dZ = np.array(dA, copy=True)
	dZ[Z <= 0] = 0
	return dZ

def linear_activation_forward(A_prev, W, B, func):
	Z = np.dot(W,A_prev) + B
	if func == 'softmax':
		A = softmax(Z)
	elif func == 'relu':
		A = np.maximum(Z,0)
	cache = (A_prev,W,B,Z)
	return A, cache

def linear_activation_backward(dA, AL, Y, cache, func):
	A_prev,W,B,Z = cache
	m = A_prev.shape[1]
	if func == 'softmax':
		dZ = AL - Y
	elif func == 'relu':
		dZ = drelu(dA, Z)
	dW = (1/m) * np.dot(dZ, A_prev.T)
	db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
	dA_prev = np.dot(W.T, dZ)
	return dA_prev, dW, db

def compute_cost(AL, Y, epsilon):
	m = Y.shape[1]
	cost = (-1/m) * np.sum(Y * np.log(AL+epsilon) + (1-Y) * np.log(1-AL+epsilon), keepdims=True, axis=1)
	cost = np.squeeze(cost)
	return cost

def update_parameters(parameters, grads, lr):
	L = len(parameters)//2
	for l in range(L):
		parameters["W" + str(l+1)] -= lr * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] -= lr * grads["db" + str(l+1)]
	return parameters

def nn_model(X, Y, X_test, Y_test, n_x, n_h, n_y, parameters):
	lr = 0.02
	epoch = 10000
	epsilon = 1e-6
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	grads = {}
	costs = []
	val_loss = []
	for i in range(0, epoch):
		A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
		A2, cache2 = linear_activation_forward(A1, W2, b2, 'softmax')
		A1_test, _ = linear_activation_forward(X_test, W1, b1, 'relu')
		A2_test, _ = linear_activation_forward(A1_test, W2, b2, 'softmax')
		cost = compute_cost(A2, Y, epsilon)
		loss = compute_cost(A2_test, Y_test, epsilon)
		dA2 = -(np.divide(Y, A2 + epsilon) - np.divide(1 - Y, 1 - A2 + epsilon))
		dA1, dW2, db2 = linear_activation_backward(dA2, A2, Y, cache2, 'softmax')
		dA0, dW1, db1 = linear_activation_backward(dA1, A2, Y, cache1, 'relu')
		grads['dW1'] = dW1
		grads['db1'] = db1
		grads['dW2'] = dW2
		grads['db2'] = db2
		parameters = update_parameters(parameters, grads, lr)
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']
		if i > 2000 and i % 100 == 0:
			lr = (1./ (1. + lr * epoch))
		if i % 100 == 0:
			print("epoch {}/{} - loss: {} - val_loss: {}".format(i+1, epoch, "%.4f" % cost, "%.4f" % loss))
			costs.append(cost)
			val_loss.append(loss)
	plt.plot(np.squeeze(costs), 'b', label="costs")
	plt.plot(np.squeeze(val_loss), 'r', label="val_loss")
	plt.ylabel('cost')
	plt.xlabel('iterations per hundreds')
	plt.title("learning rate = " + str(lr))
	plt.legend()
	plt.show()
	return parameters

def predict(X, Y, parameters):
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	A1, _ = linear_activation_forward(X, W1, b1, 'relu')
	yhat, _ = linear_activation_forward(A1, W2, b2, 'softmax')
	yhat = np.where(yhat > 0.5, 1, 0)
	result = (yhat == Y).mean()
	return result, yhat

def prediction_print(X, Y, parameters, train_bool):
	accuracy, yhat = predict(X, Y, parameters)
	print("{} accuracy on the {} set".format("%.2f" % (accuracy * 100), "training" if train_bool is True else "test"))
	tn, fp, fn, tp = confusion_matrix(np.squeeze(yhat), np.squeeze(Y)).ravel()
	print("Confusion matrice:\n  tp:{:>4} | fp:{:>4}\n  fn:{:>4} | tn:{:>4}\n".format(tp, fp, fn, tn))

if __name__ == "__main__":
	options = get_args()
	if options.mode != "prediction" and options.mode != "learning":
		print("You have to choose between learning and prediction")
		exit(0)
	if options.mode == "learning":
		X_train, X_test, Y_train, Y_test = get_data(options.data)
		n_x, n_h, n_y = layer_sizes(X_train, Y_train)
		parameters = initialize_parameters(n_x, n_h, n_y)
		parameters = nn_model(X_train, Y_train, X_test, Y_test, n_x, n_h, n_y, parameters)
		try:
			with open('parameters.pkl', 'wb') as output:
				pickle.dump(parameters, output)
		except:
			print("Something's wrong with the paremeters.pkl file")
			exit(1)
		prediction_print(X_train, Y_train, parameters, True)
		prediction_print(X_test, Y_test, parameters, False)
	elif options.mode == "prediction":
		try:
			data = pd.read_csv(options.data, header=None)
		except:
			print("This csv doesn't work")
			exit(1)
		Y = np.array([data[1]])
		Y = np.where(Y == 'M', 1, 0)
		X = np.array(data.iloc[:,2:])
		X = Normalize_data(X)
		X = X.T
		try:
			with open('parameters.pkl', 'rb') as output:
				parameters = pickle.load(output)
		except:
			print("Something's wrong with the paremeters.pkl file")
			exit(1)
		accuracy, yhat = predict(X, Y, parameters)
		print("{} accuracy on the given data set".format("%.2f" % (accuracy * 100)))
		tn, fp, fn, tp = confusion_matrix(np.squeeze(yhat), np.squeeze(Y)).ravel()
		print("Confusion matrice:\n  tp:{:>4} | fp:{:>4}\n  fn:{:>4} | tn:{:>4}\n".format(tp, fp, fn, tn))
