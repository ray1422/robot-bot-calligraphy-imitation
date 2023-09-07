"""
(c) 2023 Ray Chung. All Rights Reserved.
This program searches for the coefficients of affine transformations
that apply to robot traces to minimize the differences between our online
data and the target images.
"""

from typing import List
import cv2
import numpy as np
import scipy
import scipy.optimize

from sim import CalSimSimple, CalSimTrans3D


def search_params(sim_obj: CalSimTrans3D, target_image) -> np.ndarray:
    """
    :param path: path are 3d points with shape (n, 3)
    :param params: params are (affine on x-z plane, bias on y)
    """

    def _loss_func(params: List[float]) -> float:
        # loss is defined as IoU, but the background is white
        # so we need to invert the image
        transformed = sim_obj.transform(params)
        intersection = np.sum(
            np.logical_and(
                transformed.get_image() == 0,
                target_image == 0
            )
        )
        union = np.sum(
            np.logical_or(
                transformed.get_image() == 0,
                target_image == 0
            )
        )
        # mse = np.mean(
        #     np.square(
        #         transformed.get_image() - target_image
        #     )
        # )
        loss = 1 - intersection / union
        # loss = mse
        # transformed = sim_obj.transform(params)
        # loss = np.sum(np.abs(transformed.get_image() - target_image))
        return loss

    height = width = 256
    scale_range = (0.5, 2.0)
    grid = [
        (0, height // 4, height // 4 // 4),  # trans_x
        (0, width // 4, width // 4 // 4),  # trans_y
        (-1.57, 1.57, 1.57 / 4),  # angle from -90 deg to 90 deg but in radian
        (scale_range[0], scale_range[1],
         (scale_range[1] - scale_range[0]) / 4),  # scale_x
        (scale_range[0], scale_range[1],
         (scale_range[1] - scale_range[0]) / 4),  # scale_y
        # (-0.3, 0.3, 0.2),  # shear_x
        # (-0.3, 0.3, 0.2),  # shear_y
        (-1.0, 1.0, .8),  # height
    ]

    # split the range into 3 parts, use the best one for the next iteration
    # And then repeat the process until the range is small enough

    last_loss = np.inf
    last_result = None
    for _ in range(10):
        result, loss, _, _ = scipy.optimize.brute(
            _loss_func,
            grid,
            full_output=True
        )
        print(result, loss)

        if loss < last_loss:
            last_loss = loss
            for j in range(len(grid)):
                delta = (grid[j][2] / 2)
                grid[j] = [result[j] - delta, result[j] + delta, delta * 2 / 3]
            last_result = result.copy()

        else:
            return last_result


if __name__ == '__main__':
    cal_sim = CalSimSimple(file='test_strokes/char00482_stroke.txt')
    sims = cal_sim.split_strokes()
    # just test the first stroke
    sim: CalSimTrans3D = CalSimTrans3D.from_cal_sim_simple(sims[1])
    from_img = sim.get_image()
    cv2.imwrite('from_img.png', from_img)
    target_image = cv2.imread('test_strokes/tmp1_1.jpg', cv2.IMREAD_GRAYSCALE)

    params = search_params(sim, target_image)

    result = sim.transform(params)
    result_img = result.get_image()
    cv2.imwrite('result_img.png', result_img)

    print(params)
