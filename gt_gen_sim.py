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
import matplotlib.pyplot as plt

from sim import CalSimSimple, CalSimTrans3D


def search_params(sim_obj: CalSimTrans3D, target_image) -> (np.ndarray, float):
    """
    :param path: path are 3d points with shape (n, 3)
    :param params: params are (affine on x-z plane, bias on y)
    :return: params and loss
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

    # first of all, align the center
    from_img = sim_obj.get_image()
    _, th = cv2.threshold(from_img, 0, 32, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # find bounding box
    # take the largest connected component
    cnts, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cnt_1 = max(cnts, key=cv2.contourArea)
    _, th = cv2.threshold(target_image, 0, 32, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cnt_2 = max(cnts, key=cv2.contourArea)

    # find the center of the bounding box
    x1, y1, w1, h1 = cv2.boundingRect(cnt_1)
    x2, y2, w2, h2 = cv2.boundingRect(cnt_2)

        
    center_1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
    center_2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
    # calculate the translation
    scale_x = w2 / w1
    scale_y = h2 / h1
    rotate = 0.
    loss_min = np.inf
    rotate = cv2.ROTATE_90_CLOCKWISE
    if scale_x / scale_y > 3 or scale_y / scale_x > 3:
        # test rotate 90 degree clockwise or counter-clockwise
        crop_from = from_img[y1:y1+h1, x1:x1+w1].copy()
        for deg in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            crop_from_rotated = cv2.rotate(crop_from.copy(), deg)
            crop_from_rotated = cv2.resize(crop_from_rotated, (w2, h2))
            target_image_crop = target_image[y2:y2+h2, x2:x2+w2].copy()
            # loss is defined as IoU, but the background is white and the stroke is black
            intersection = np.sum(
                np.logical_and(
                    crop_from_rotated == 0,
                    target_image_crop == 0
                )
            )
            union = np.sum(
                np.logical_or(
                    crop_from_rotated == 0,
                    target_image_crop == 0
                )
            )
            loss = 1 - intersection / union
            # print(loss)
            if loss < loss_min:
                loss_min = loss
                from_img = crop_from_rotated
                rotate = deg
        # calculate the translation
        scale_x = w2 / h1
        scale_y = h2 / w1
        rotate = 1.57 if rotate == cv2.ROTATE_90_CLOCKWISE else -1.57 # y axis is flipped so the rotation is flipped
        # calculate new center after rotation
        # maps to polar coordinate
        
        r = np.sqrt(center_1[0] ** 2 + center_1[1] ** 2)
        theta = np.arctan2(center_1[1], center_1[0])
        # rotate
        theta += rotate
        # maps back to cartesian coordinate
        center_1 = np.array([r * np.cos(theta), r * np.sin(theta)])
 

    
    shift_x = center_2[0] - center_1[0] * scale_x
    shift_y = center_2[1] - center_1[1] * scale_y
 

    # save initial guess
    sim2 = sim_obj.transform([shift_x, shift_y, rotate, scale_x, scale_y, 0])
    from_img = sim2.get_image()
    cv2.imwrite('from_img.png', from_img)
    
    # print(shift_x, shift_y, scale_x, scale_y, rotate)
    height = width = 256
    scale_range = (0.5, 2.0)
    grid = [
        (shift_x - height // 8, shift_x+height // 8, height // 4 // 4),  # trans_x
        (shift_y - width // 9, shift_y + width // 8, width // 4 // 4),  # trans_y
        (-1.57, 1.57, 1.57 / 8),  # angle from -90 deg to 90 deg but in radian
        # (scale_range[0], scale_range[1],
        #  (scale_range[1] - scale_range[0]) / 4),  # scale_x
        # (scale_range[0], scale_range[1],
        #  (scale_range[1] - scale_range[0]) / 4),  # scale_y
        (scale_x * .8, scale_x * 1.25, 0.15),  # scale_x
        (scale_y * .8, scale_y * 1.25, 0.15),  # scale_y
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
        # print(result, loss)

        if loss < last_loss:
            last_loss = loss
            for j in range(len(grid)):
                delta = (grid[j][2] / 2)
                grid[j] = [result[j] - delta, result[j] + delta, delta * 2 / 3]
            last_result = result.copy()

        else:
            return last_result, last_loss
    
    return last_result, last_loss

if __name__ == '__main__':
    cal_sim = CalSimSimple(file='sample_data/char00615_stroke.txt')
    sims = cal_sim.split_strokes()
    # just test the first stroke
    sim: CalSimTrans3D = CalSimTrans3D.from_cal_sim_simple(sims[0])
    from_img = sim.get_image()
    
    #cv2.imwrite('from_img.png', from_img)
    target_image = cv2.imread('sample_data/tmp1_0.jpg', cv2.IMREAD_GRAYSCALE)

    params, loss = search_params(sim, target_image)

    result = sim.transform(params)
    result_img = result.get_image()
    #cv2.imwrite('result_img.png', result_img)
    print(params)

    # show in image
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(from_img, cmap="gray")
    ax1.set_title("x")
    ax2 = fig.add_subplot(1, 3, 2, sharey=ax1, sharex=ax1)
    ax2.imshow(target_image, cmap="gray")
    ax2.set_title("y")
    ax3 = fig.add_subplot(1, 3, 3, sharey=ax1, sharex=ax1)
    ax3.imshow(result_img, cmap="gray")
    ax3.set_title("y'")
    # plt.show()
    plt.savefig("output.png")


    #寫到txt file利用機器手臂寫出
    value_3d = result.trace_3d  #這是原本0~256邊界的
    #print("調整前")
    #print(value_3d)
    
    # Call inverse_export_to_robot to adjust the result.trace_3d
    ret = result.export_to_robot()
    print(ret)
    exit()
    output_file = "output.txt"
    min = 1000
    for item in value_3d:
        data_3D_rounded = [round(value, 4) for value in item]
        if int(data_3D_rounded[2]) < min:
            min = int(data_3D_rounded[2])
    last_add = 173 - min
    with open(output_file, "w") as file:
        for item in value_3d:
            data_3D_rounded = [round(value, 4) for value in item]
            data_3D_rounded[2] += last_add
            formatted_item = " ".join(map(str, data_3D_rounded))
            output_line = f"movl 0 {formatted_item} -180 0 40 100.0000 stroke1"
            file.write(output_line + "\n")
    print("File writing completed.")


