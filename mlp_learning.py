import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
	if (len(sys.argv) != 2):
		print("Give a valid data.csv")
		exit(1)

	data = pd.read_csv("data.csv")
	print(data)
