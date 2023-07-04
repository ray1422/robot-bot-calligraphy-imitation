import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATA_PATH = "result/SG_thin"
SAVE_PATH = "result/SG_extreme"

def load_data(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img = np.array(img)
    img = binarization(img)
    return img

def make_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)

def binarization(img):
    thres = 128
    for i in range(256):
        for j in range(256):
            if img[i, j] < thres:
                img[i, j] = 0
            else:
                img[i, j] = 255
    return img 

def has_value(i, j, skeleton):
        # if skeleton[i, j] == 1:
        if skeleton[i, j] != 0:
            return True
        return False

def is_extreme(i, j, skeleton):
    # start/end point or not
    # e.g.
    # 0  1  0
    # 0  1  0
    # 0  0  0
    if has_value(i, j, skeleton):
        a = update_point(i, j, skeleton)
        if np.count_nonzero(a) == 1:
            return True
    return False

def is_connected(i, j, skeleton):
    # connected point or not
    # e.g.
    #  0  1  0
    #  0  1  1
    #  0  1  0
    if has_value(i, j, skeleton):
        a = update_point(i, j, skeleton)
        if np.count_nonzero(a) > 3:
            return True
    return False

def update_point(i, j, skeleton):
    new_point = [skeleton[i + 1, j    ], 
                skeleton[i + 1, j + 1], 
                skeleton[i    , j + 1], 
                skeleton[i - 1, j + 1],
                skeleton[i - 1, j    ], 
                skeleton[i - 1, j - 1], 
                skeleton[i    , j - 1], 
                skeleton[i + 1, j - 1]]
    return new_point

def detect_keypoint(path):
    # thinned image
    skeleton = load_data(path)
    extreme = []
    connect = []
    for i in range(256):
        for j in range(256):
            if is_extreme(i, j, skeleton):
                skeleton[i, j] = 128
                extreme.append((i,j))
                continue
            if is_connected(i, j, skeleton):
                skeleton[i, j] = 168
                connect.append((i,j))
    # show
    plt.imshow(skeleton, cmap='hot') # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.savefig(SAVE_PATH + '/' + path[-5] + '/' + 'SG_extreme_%s.jpg' % path[-5])
    # plt.show()
    np.savetxt(SAVE_PATH + '/' + path[-5] + '/' + 'SG_keypoint_%s.txt' % path[-5], skeleton, fmt='%3d')
    np.savetxt(SAVE_PATH + '/' + path[-5] + '/' + 'SG_extreme_%s.txt' % path[-5], extreme, fmt='%3d')
    np.savetxt(SAVE_PATH + '/' + path[-5] + '/' + 'SG_connect_%s.txt' % path[-5], connect, fmt='%3d')

if __name__ == '__main__':
    for root, dirs, fs in os.walk(DATA_PATH):
        for f in fs:
            if len(f) == 13:
                p = os.path.join(root, f)
                make_dir(SAVE_PATH + '/' + f[-5])
                print(p)
                detect_keypoint(p)