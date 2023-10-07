import cv2
import numpy as np
import torch


def affine_pytorch2warp(theta, width, height):
    # add a row [0, 0, 1] to theta
    theta = np.concatenate([theta, np.array([[0, 0, 1]])], axis=0)
    s = np.array([
        [2./(width-1), 0, -1],
        [0, 2./(height-1), -1],
        [0, 0, 1]
    ], dtype=np.float32)
    mat_a = np.linalg.pinv(s)
    mat_b = np.linalg.pinv(theta)
    mat_c = s
    

    mat = mat_a @ mat_b @ mat_c

    return mat[:2, :]


def affine_warp2pytorch(mat: np.ndarray, width=256, height=256):
    """
    :param mat: 2x3 affine matrix
    """

    # add a row [0, 0, 1] to mat
    mat = np.concatenate([mat, np.array([[0, 0, 1]])], axis=0)
    # convert to pytorch affine grid
    mat_a = np.array([
        [2./(width-1), 0, -1],
        [0, 2./(height-1), -1],
        [0, 0, 1]
    ], dtype=np.float32)
    mat_b = np.linalg.pinv(mat)
    mat_c = np.linalg.pinv(mat_a)

    theta = mat_a @ mat_b @ mat_c

    return theta[:2, :]


def test_affine_pytorch2warp():
    sample_images = np.random.randint(0, 256, (10, 256, 256), dtype=np.uint8)
    # gussian blur
    for i, img in enumerate(sample_images):
        sample_images[i] = cv2.GaussianBlur(img, (31, 31), 21)
    for i in range(10):
        theta = np.random.rand(2, 3) * 2. - 1
        mat = affine_pytorch2warp(theta, 256, 256)
        theta = torch.from_numpy(theta).float()
        img_torch = torch.from_numpy(sample_images[i:i+1]).float().reshape((1, 1, 256, 256))
        img = sample_images[i].reshape((256, 256))
        grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), (1, 1, 256, 256), align_corners=True)
        transformed = torch.nn.functional.grid_sample(img_torch, grid, align_corners=True)
        transformed_warp = cv2.warpAffine(img, mat, (256, 256))
        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 3, 1)
        # plt.imshow(img)
        # plt.subplot(1, 3, 2)
        # plt.imshow(transformed.numpy()[0, 0])
        # plt.subplot(1, 3, 3)
        # plt.imshow(transformed_warp)
        # plt.show()
        print(np.abs(transformed.numpy()[0, 0] - transformed_warp.reshape((256, 256))).mean())

        assert np.abs(transformed.numpy()[0, 0] - transformed_warp.reshape((256, 256))).mean() < .2
