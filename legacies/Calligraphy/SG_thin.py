import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize

IMG_PATH = "data/SG"
SAVE_PATH = "result"

def binarization(img):
    thres = 128
    img[np.where(img <= thres)] = 1
    img[np.where(img > thres)] = 0
    return img

def de_binary(img):
    img[np.where(img > 0)] = 255
    return img

def load_data(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256), Image.ANTIALIAS)
    gray = img.convert('L')
    gray = np.array(gray)
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
    blur_gray = np.array(blur_gray)
    blur_gray = binarization(blur_gray)
    return blur_gray

def thin(img):
    skeleton = skeletonize(img).astype(np.uint8)
    return skeleton

def dil_ero(img):
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    return erosion

def closing(img):
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing
    
def make_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)

def neighbours(x, y, image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]  # P2, P3, ... , P8, P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1  # the points to be removed (set as 0)
    while changing1 or changing2:  # iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns  = Image_Thinned.shape  # x for rows, y for columns
        for x in range(1, rows - 1):  # No. of  rows
            for y in range(1, columns - 1):  # No. of columns
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions 
                        2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                        transitions(n) == 1 and  # Condition 2: S(P1)=1  
                        P2 * P4 * P6 == 0 and  # Condition 3   
                        P4 * P6 * P8 == 0):  # Condition 4
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0
                        2 <= sum(n) <= 6 and  # Condition 1
                        transitions(n) == 1 and  # Condition 2
                        P2 * P4 * P8 == 0 and  # Condition 3
                        P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


if __name__ == '__main__':
    for root, dirs, fs in os.walk(IMG_PATH):
        for f in fs:
            p = os.path.join(root, f)
            img = load_data(p)
            img = zhangSuen(img)
            img[img > 0] = 255
            # img = thin(img)
            # img = dil_ero(img)      
            # img = thin(img)
            # img = closing(img)
            # img = de_binary(img)
            # cv2.imshow('closing img', img)
            save_name = 'SG_thin_%s.jpg' % root[-1]
            print(save_name)
            make_dir(SAVE_PATH + '/' + 'SG_thin')
            make_dir(SAVE_PATH + '/' + 'SG_thin' + '/' + root[-1])
            cv2.imencode('.jpg', img)[1].tofile('result/SG_thin' + '/' + root[-1] + '/' + save_name)
            # cv2.waitKey(200)