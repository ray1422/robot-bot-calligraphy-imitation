"""
(c) 2023 Ray Chung. All Rights Reserved.
This program searches for the coefficients of affine transformations 
between calibration images to minimize the differences between styles.
"""
import logging
import os
import traceback
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import scipy.optimize
import matplotlib.pyplot as plt

# use gtk backend for matplotlib
import matplotlib


# matplotlib.use('TKAgg')


def my_transform(x: np.array, params: List[float]) -> np.array:
    """
    :param x: input image with type array of np.float32
    :param params: (trans_x, trans_y, angle, scale_x, scale_y, shear_x, shear_y)
    """

    # parameters for affine transformation
    trans_x, trans_y, angle, scale_x, scale_y, shear_x, shear_y = params

    # rotation angle in radian, scale in percentage
    m_rotate = np.asarray([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0]
    ], dtype=float)

    # translation in pixel
    m_trans = np.asarray([
        [1, 0, trans_x],
        [0, 1, trans_y],
        [0, 0, 1]
    ], dtype=float)
    m_scale = np.asarray([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=float)
    m_shear = np.asarray([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ], dtype=float)
    affine_param = m_rotate @ m_trans @ m_scale @ m_shear  # TODO optimize
    # padding with 0
    y_ = cv2.warpAffine(x, affine_param[:2, :], (x.shape[1], x.shape[0]),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return y_


def enhance_image(x: np.ndarray) -> np.ndarray:
    """
    :param x: input image with type array of np.float32
    :return: enhanced image with type array of np.float32
    """
    x = x.copy()
    # normalize to [0, 1] and invert color and convert to gray scale
    x = 255 - x
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = np.asarray(x, dtype=np.float32) / 255.0
    # TODO enhance image
    return x


def find_stroke_transform(x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """
    `find_stroke_transform` finds coefficient of affine transformation 
    of two styles images of the same stroke of calibration image.
    The returned transformation can converts x to y' and minimize
    L-2 norm of y - y'.
    param x: photo a, image with type array of np.float32
    param y: input image with type array of np.float32
    return: optimized coefficient of affine transformation. None if failed.
    """

    def _loss_func(params) -> float:
        """
        :param x: src image with type array of np.float32, normalized to [0, 1]
        and background is 0
        :param y: target image
        :param params: (trans_x, trans_y, angle, scale_x, scale_y, shear_x, shear_y)
        :return: MSE of y and y'
        """
        y_ = my_transform(x.copy(), params)
        y_ = cv2.resize(y_, (y.shape[1], y.shape[0]))
        return float(np.mean((y - y_) ** 2))

    # find coefficient of affine transformation using Ternary Search
    EPS = 7e-2  # acceptable error
    w, h = x.shape
    # angle is in degree, and scale is in percentage=
    scale_range = (0.5, 2.0)
    height, width = x.shape
    # brute force search for approximate solution
    # it's like binary search but in 3 dimensions
    grid = [
        (0, height // 4, height // 4 // 3),  # trans_x
        (0, width // 4, width // 4 // 3),  # trans_y
        (-0.52, 0.52, 0.34),  # angle from -30 deg to 30 deg but in radian
        (scale_range[0], scale_range[1],
         (scale_range[1] - scale_range[0]) / 3),  # scale_x
        (scale_range[0], scale_range[1],
         (scale_range[1] - scale_range[0]) / 3),  # scale_y
        (-0.3, 0.3, 0.2),  # shear_x
        (-0.3, 0.3, 0.2),  # shear_y

    ]
    result = None
    last_loss = np.inf
    for step in range(3):
        try:
            _result, loss, _, _ = scipy.optimize.brute(
                _loss_func,
                grid,
                full_output=True
            )

            logging.debug(f"step {step} loss {loss: 3.2e}")
            if loss > last_loss:
                pass
                logging.debug("loss increased, early stop")
                break
            last_loss = loss
            result = _result
            logging.debug(result)

            # search in smaller range with more precision
            for i, item in enumerate(result.tolist()):
                grid[i] = (item - 0.5 * grid[i][2],
                           item + 0.5 * grid[i][2],
                           (grid[i][2] / 3))

        except ValueError as e:
            traceback.print_exc()
            break

    return result if last_loss < EPS else None


def gt_gen_test():
    img_a = enhance_image(cv2.imread("sample_data/stroke_d.png"))
    img_b = enhance_image(cv2.imread("sample_data/stroke_c.png"))
    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    ret = find_stroke_transform(img_a, img_b)
    if ret is None:
        print("cannot find stroke transform")
        return
    a_prime = my_transform(img_a, ret)
    # plot those images with matplotlib
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_a, cmap="gray")
    ax1.set_title("x")
    ax2 = fig.add_subplot(1, 3, 2, sharey=ax1, sharex=ax1)
    ax2.imshow(img_b, cmap="gray")
    ax2.set_title("y")
    ax3 = fig.add_subplot(1, 3, 3, sharey=ax1, sharex=ax1)
    ax3.imshow(a_prime, cmap="gray")
    ax3.set_title("y'")
    # plt.show()
    plt.savefig("output.png")


def main():
    dataset_dir = "./datasets/stroke_new"
    styles_prefix = ("s0", "s1")
    for char_name in os.listdir(dataset_dir):
        char_dir = os.path.join(dataset_dir, char_name)

        for i in range(0, 30):
            # if both s{i}a.png and s0_{i} and s1_{i} exists, calculate transform
            images = []
            for style_prefix in styles_prefix:
                img_path = os.path.join(char_dir, f"{style_prefix}_{i}.png")
                if os.path.exists(img_path):
                    # openCV can't deal with unicode path, so use numpy to read image
                    img = cv2.imdecode(np.fromfile(
                        img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    images.append(img)
            if len(images) == 0:
                break
            if len(images) != len(styles_prefix):
                # invalid data
                print(f"invalid data {char_name} {i}")
                continue
            img_a = enhance_image(images[0])
            img_b = enhance_image(images[1])
            try:
                ret = find_stroke_transform(img_a, img_b)
            except Exception as e:
                traceback.print_exc()
                break
            if ret is None:
                print(f"cannot find stroke transform {char_name} {i}")
                break
            # save transform as numpy array
            np.save(os.path.join(char_dir, f"transform_{i}.npy"), ret)
            print(f"found transform {char_name} {i}")


if __name__ == "__main__":
    main()
