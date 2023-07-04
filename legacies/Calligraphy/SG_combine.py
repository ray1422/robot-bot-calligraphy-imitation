import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cosine

DATA_PATH = "result/SG_split"
SAVE_PATH = "result/SG_combine"

def load_data(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img = np.array(img)
    # img = binarization(img)
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
                img[i, j] = 1
    return img

def cos(v_a, v_b):
    return cosine(v_a, v_b)

def is_near(paa, pab, pba, pbb, name, n1, n2):

    dis_1 = ((paa[1] - pba[1])**2 + (paa[2] - pba[2])**2)**0.5
    dis_2 = ((paa[1] - pbb[1])**2 + (paa[2] - pbb[2])**2)**0.5
    dis_3 = ((pab[1] - pba[1])**2 + (pab[2] - pba[2])**2)**0.5
    dis_4 = ((pab[1] - pbb[1])**2 + (pab[2] - pbb[2])**2)**0.5
    dis = [dis_1, dis_2, dis_3, dis_4]
    v1 = 0
    v2 = 0
    cos = 100

    if min(dis) <= 10:
        if paa[1] > pab[1]:
            v1 = (paa[1] - pab[1]), (paa[2] - pab[2])    
        else:
            v1 = (pab[1] - paa[1]), (pab[2] - paa[2])
        if paa[1] > pba[1]:
            v2 = (paa[1] - pba[1]), (paa[2] - pba[2])
        else:
            v2 = (pba[1] - paa[1]), (pba[2] - paa[2])
        if paa[1] > pbb[1]:
            v3 = (paa[1] - pbb[1]), (paa[2] - pbb[2])
        else:
            v3 = (pbb[1] - paa[1]), (pbb[2] - paa[2])
        if pab[1] > pba[1]:
            v4 = (pab[1] - pba[1]), (pab[2] - pba[2])
        else:
            v4 = (pba[1] - pab[1]), (pba[2] - pab[2])
        if pab[1] > pbb[1]:
            v5 = (pab[1] - pbb[1]), (pab[2] - pbb[2])
        else:
            v5 = (pbb[1] - pab[1]), (pbb[2] - pab[2])
        # cos = cosine(v1, v2)
        cos_1 = cosine(v1, v2)
        cos_2 = cosine(v1, v3)
        cos_3 = cosine(v1, v4)
        cos_4 = cosine(v1, v5)
        cos = max([cos_1, cos_2, cos_3, cos_4])
        #if name == '永':
        print('n1: %s, n2: %s' % (n1+1, n2+1))
        print('cos:', cos)
        if cos < 0.1:
            #if name == '永':
            print('Combine.')
            return True    
    return False

def combine(p):
    name = p[-1]
    v = []
    txt = np.loadtxt(DATA_PATH + '/%s' % name + '/%s_start_end.txt' % name, delimiter=',')   
    # print(txt[-1][0])
    # 計算所有筆畫的方向
    for i in range(int(txt[-1][0])):
        a = txt[(3 * i)]
        b = txt[(2 + 3 * i)]
        if a[1] > b[1]:
            v_ = (a[2] - b[2]), (a[1] - b[1])
            v.append(v_)
        else:
            v_ = (b[2] - a[2]) , (b[1] - a[1])
            v.append(v_)
    
    t = np.zeros(int(txt[-1][0]))
    count = 1
    for i in range(int(txt[-1][0])):
        for j in range(i+1, int(txt[-1][0])):
            cos_dis = cos(v[i], v[j])
            print('%s | i: %d, j: %d, cos: %f' % (name, i+1, j+1, cos_dis))
            if cos_dis < 0.1 and is_near(txt[(3 * i)], txt[(2 + 3 * i)], txt[(3 * j)], txt[(2 + 3 * j)], name, i, j):
                print(cos_dis)
                # connect.append([i+1, j+1])
                if t[i] != 0:
                    t[j] = t[i]
                elif t[j] != 0:
                    t[i] = t[j]
                else:
                    t[i] = count
                    t[j] = count
                    count = count + 1
    print(name)
    print(t)
    make_dir(SAVE_PATH + '/' + name)
    order = 1
    for i in range(int(max(t) + 1)):
        stroke_img = np.zeros((256, 256), dtype=np.int16)
        for j in range(len(t)):
            if t[j] == 0:
                img = load_data(p + '/SG_%s_%02d.jpg' % (name, j+1))
                save_name = 'SG_combine_%s_%02d.jpg' % (name, order)
                cv2.imencode('.jpg', img)[1].tofile(SAVE_PATH + '/' + name + '/' + save_name)
                order = order + 1
                t[j] = -1
                continue
            if t[j] == i:
                img = load_data(p + '/SG_%s_%02d.jpg' % (name, j+1))
                stroke_img = stroke_img + img
                stroke_img[np.where(stroke_img > 255)] = 255
        save_name = 'SG_combine_%s_%02d.jpg' % (name, order)
        cv2.imencode('.jpg', stroke_img)[1].tofile(SAVE_PATH + '/' + name + '/' + save_name)
        order = order + 1


if __name__ == '__main__':
       
    dirs = os.listdir(DATA_PATH)
    # print(dirs)
    make_dir(SAVE_PATH)
    for d in dirs:
        p = DATA_PATH + '/' + d
        combine(p)
            
                
                