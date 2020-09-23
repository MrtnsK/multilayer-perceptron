import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def LR_descent_gradient(X, Y):
	Lrate = 1
	thetas = np.array([0,0])
	m = float(len(X))

	while (1):
		tmp = thetas
		thetas = thetas - Lrate * (1 / m) * (X.T @ ((X @ thetas) - Y))
		if np.array_equal(thetas, tmp):
			break
	return thetas

if __name__ == "__main__":
	if (len(sys.argv) != 2):
		print("Give a valid data.csv")
		exit(1)

	data = pd.read_csv("data.csv")
	print(data)
