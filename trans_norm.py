# load all transforms in dataset/stroke/*.npy

import glob
import numpy as np


data = [np.load(npy_name) for npy_name in glob.glob("./datasets/sim/*/transform.npy")]
masks = [np.load(npy_name) for npy_name in glob.glob("./datasets/sim/*/mask.npy")]
filter_data = np.zeros((0, 6), dtype=np.float32)
for i, d in enumerate(data):
    for j, u in enumerate(d):
        if masks[i][j][0]:
            filter_data = np.append(filter_data, [u], axis=0)
            


error = 0
cnt = 0


mean = np.mean(filter_data, axis=0)
std = np.std(filter_data, axis=0)
print("mean:", mean)
print("std:", std)







# calculate std and mean of produced transform
def params2mat(params):
    trans_x, trans_y, angle, scale_x, scale_y, z_bias = params
    # rotation angle in radian, scale in percentage
    m_rotate = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=float,
    )

    # translation in pixel
    m_trans = np.asarray([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]], dtype=float)
    m_scale = np.asarray([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=float)
    # m_shear = np.asarray([
    #     [1, shear_x, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ], dtype=float)
    affine_param = m_trans @ m_scale @ m_rotate  # NOTE: the order is important
    return affine_param


cnt = 0
mats = np.zeros((len(filter_data), 2, 3), dtype=np.float32)
for i, d in enumerate(data):
    for j, u in enumerate(d):
        if masks[i][j][0]:
            mats[cnt, :, :] = params2mat(u)[:2, :]
            cnt += 1

mean = np.mean(mats, axis=0)
std = np.std(mats, axis=0)

print("mat mean:", mean)
print("mat std:", std)


# test error if use means as prediction
error = 0
cnt = 0
for i, d in enumerate(data):
    for j, u in enumerate(d):
        if masks[i][j][0]:
            cnt += 1
            normalized = (params2mat(u)[:2, :] - mean[:2, :]) / std[:2, :]
            error += np.mean(np.square(normalized))
error = error / cnt
print("error:", error)
