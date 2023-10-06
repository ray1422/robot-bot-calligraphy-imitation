# load all transforms in dataset/stroke/*.npy

import glob
import numpy as np


data = np.asarray([np.load(npy_name) for npy_name in glob.glob("./datasets/stroke/*/mat_*.npy")])

# data.shape = (N, 9)
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)
print(means)
print(stds)

data_2 = (data[:, :2, :] - means[:2, :]) / stds[:2, :]

# calculate error to mean
error = np.mean(np.square(data_2))
print(error)
