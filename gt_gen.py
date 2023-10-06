"""
(c) 2023 Ray Chung. All Rights Reserved.
This program searches for the coefficients of affine transformations 
between calibration images to minimize the differences between styles.
"""
import logging
from multiprocessing import Pool
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


def find_bounding_box(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    find bounding box of stroke
    :param img: input image with type array of 0-255
    :return: bounding box (x, y, w, h)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # coords = cv2.findNonZero(th)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # use the largest contour
    coords = max(cnts[0], key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def params2mat(params: List[float]) -> np.ndarray:
    trans_x, trans_y, angle, scale_x, scale_y, shear_x, shear_y = params

    # rotation angle in radian, scale in percentage
    m_rotate = np.asarray([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
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
    return affine_param


def my_transform(x: np.array, params: List[float]) -> np.array:
    """
    :param x: input image with type array of np.float32
    :param params: (trans_x, trans_y, angle, scale_x, scale_y, shear_x, shear_y)
    """
    x = 1 - x
    # parameters for affine transformation
    affine_param = params2mat(params)
    # padding with 0
    y_ = cv2.warpAffine(x, affine_param[:2, :], (x.shape[1], x.shape[0]),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    y_ = 1 - y_
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

    # first find bounding box of stroke and generate initial guess
    x1, y1, w1, h1 = find_bounding_box(x)
    # cv2.imshow("x", x)
    # cv2.waitKey(0)
    x2, y2, w2, h2 = find_bounding_box(y)
    # cv2.imshow("y", y)
    # cv2.waitKey(0)
    # m_rotate @ m_trans @ m_scale @ m_shear
    # scale applied first, then rotate, then translate
    # initial guess
    scale_x = w2 / w1
    scale_y = h2 / h1
    angle = 0
    trans_x = x2 - x1 * scale_x
    trans_y = y2 - y1 * scale_y
    shear_x = 0
    shear_y = 0

    x_crop = x[y1:y1 + h1, x1:x1 + w1, 0] / 255.0
    y_crop = y[y2:y2 + h2, x2:x2 + w2, 0] / 255.0
    x_crop = cv2.resize(x_crop, (y_crop.shape[1], y_crop.shape[0]))
    # pads 2x size of image with 1
    x_crop = np.pad(x_crop, ((y_crop.shape[0], y_crop.shape[0]), (y_crop.shape[1], y_crop.shape[1])),
                    mode="constant", constant_values=1)
    y_crop = np.pad(y_crop, ((y_crop.shape[0], y_crop.shape[0]), (y_crop.shape[1], y_crop.shape[1])),
                    mode="constant", constant_values=1)
    mat_1 = params2mat([trans_x, trans_y, angle, scale_x, scale_y, shear_x, shear_y])

    # keep initial guess, crop, and combine it with brute force search later

    def _loss_func(params) -> float:
        """
        :param x: src image with type array of np.float32, normalized to [0, 1]
        and background is 0
        :param y: target image
        :param params: (trans_x, trans_y, angle, scale_x, scale_y, shear_x, shear_y)
        :return: MSE of y and y'
        """
        y_ = my_transform(x_crop.copy(), params)
        # y_ = cv2.resize(y_, (y.shape[1], y.shape[0]))
        assert y_crop.shape == y_.shape
        # use IoU as loss function
        # but the background is 1, and foreground is 0
        th = 0.5

        intersection = np.logical_and(y_crop < th, y_ < th)
        union = np.logical_or(y_crop < th, y_ < th)
        iou_score = np.sum(intersection) / np.sum(union)
        return 1 - iou_score

    # find coefficient of affine transformation using Ternary Search
    EPS = 0.3  # acceptable error
    w, h = x_crop.shape
    # angle is in degree, and scale is in percentage=
    scale_range = (0.5, 2.0)
    height, width = x_crop.shape
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
    result = [0, 0, 0, 1., 1., 0, 0]
    # use initial guess to find approximate solution as worst case
    last_loss = _loss_func((0, 0, 0, 1., 1., 0, 0))
    # print("last_loss", last_loss)
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

    if last_loss > EPS:
        # failed to find solution
        return None

    # combine initial guess and result
    # we first apply initial guess, then apply result

    mat_2 = params2mat(result)

    mat = mat_2 @ mat_1

    # visualize
    y2 = 255 - x
    y2 = cv2.warpPerspective(y2, mat[:, :], (x.shape[1], x.shape[0]),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    y2 = 255 - y2
    # plot those images with matplotlib
    # matplotlib.use('TKAgg')
    # ax = plt.subplot(1, 3, 1)
    # ax.imshow(x, cmap="gray")
    # ax.set_title("x")
    # ax = plt.subplot(1, 3, 2, sharey=ax, sharex=ax)
    # ax.imshow(y, cmap="gray")
    # ax.set_title("y")
    # ax = plt.subplot(1, 3, 3, sharey=ax, sharex=ax)
    # ax.imshow(y2, cmap="gray")
    # ax.set_title("y'")
    # plt.show()

    return mat


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


def job(img_a, img_b, char_name, i, char_dir):
    try:
        ret = find_stroke_transform(img_a, img_b)
    except Exception as e:
        traceback.print_exc()
        return None
    if ret is None:
        print(f"cannot find stroke transform {char_name} {i}")
        return None
    # save transform as numpy array
    np.save(os.path.join(char_dir, f"transform_{i}.npy"), ret)
    print(f"found transform {char_name} {i}")


def main():
    dataset_dir = "./datasets/stroke"
    styles_prefix = ("s0", "s1")
    with Pool(8) as p:
        results = []
        for char_name in os.listdir(dataset_dir):
            char_dir = os.path.join(dataset_dir, char_name)
            # print(char_dir)
            for i in range(0, 30):
                # if both s{i}a. and s0_{i} and s1_{i} exists, calculate transform
                images = []
                for style_prefix in styles_prefix:
                    img_path = os.path.join(char_dir, f"{style_prefix}_{i}_full.png")

                    if os.path.exists(img_path):
                        # openCV can't deal with unicode path, so use numpy to read image
                        img = cv2.imread(img_path)
                        images.append(img)
                if len(images) == 0:
                    break
                if len(images) != len(styles_prefix):
                    # invalid data
                    print(f"invalid data {char_name} {i}")
                    continue
                # img_a = enhance_image(images[0])
                # img_b = enhance_image(images[1])
                img_a = images[0]
                img_b = images[1]
                ret = p.apply_async(job, (img_a, img_b, char_name, i, char_dir))
                results.append(ret)
        for ret in results:
            ret.get()


if __name__ == "__main__":
    main()
