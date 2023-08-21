import cv2
import numpy as np
img_a = cv2.imread("./sample_data/stroke_a.png")
img_b = cv2.imread("./sample_data/stroke_b.png")
img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
x = 255 - img_a.copy()
trans_x, trans_y, angle, scale_x, scale_y = 10, -10, np.pi/4, 1.5, 0.8
# parameters for affine transformation

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
affine_param = m_rotate @ m_trans @ m_scale  # TODO optimize
y_ = cv2.warpAffine(x, affine_param[:2, :], (x.shape[1], x.shape[0]))
cv2.imshow("asdf", x)
cv2.waitKey(0)
cv2.imshow("asdf", y_)
cv2.waitKey(0)
print(affine_param)
