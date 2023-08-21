# load all transforms in dataset/stroke/*.npy

import glob
import numpy as np


data = np.asarray([np.load(npy_name) for npy_name in glob.glob("./datasets/stroke_new/*/*.npy")])

# data.shape = (N, 7)
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)
print(means)
print(stds)

data_2 = (data - means) / stds
# calculate error to mean
error = np.mean((data_2 - means) ** 2)
print(error)