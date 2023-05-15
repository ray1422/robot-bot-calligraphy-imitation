
import math
from typing import List, Tuple
import numpy as np
import cv2


def enhance(img: np.ndarray) -> np.ndarray:
    """
    params img: input image
    return: enhanced image in grayscale
    """
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img2 = cv2.resize(img2, (256, 256))
    ret, th2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY_INV)
    # erode and dilation
    kernel = np.ones((3, 3), np.uint8)
    th2 = cv2.erode(th2, kernel, iterations=2)
    th2 = cv2.dilate(th2, kernel, iterations=2)

    return th2


def sub_contour(img: np.ndarray) -> List[np.ndarray]:
    """
    param img: grayscale enhanced image
    """
    n, objs = cv2.connectedComponents(img)
    ret = []
    print(objs.shape)
    for i in range(n):
        mask = np.asarray(objs == i, dtype=np.uint8) * 255
        # remove background. 255 is threshold
        if np.sum(np.bitwise_and(mask, img)) < 255:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        # should apply bounding check in the future.
        ret.append(mask[y-5:y+h+5, x-5:x+w+5])
    return ret


def edge_detection(img: np.ndarray) -> np.ndarray:
    """
    img: grayscale img of a part
    return: edges
    """
    edge = cv2.Canny(img, 100, 200)
    return edge


def vertex_extract_and_smooth(img: np.ndarray) -> np.ndarray:
    """
    img: edges
    returns: sequence of points
    """
    vs = []
    h, w = np.shape(img)
    for i in range(h):
        for j in range(w):
            if img[i, j] > 128:
                vs.append((i, j))

    def to_polar(a: Tuple[int, int]) -> Tuple[float, float]:
        r = math.sqrt((a[0] - h)**2 + (a[1]-w) ** 2)
        theta = math.atan2(a[0], a[1])
        return float(r), float(theta)
    # sorts
    theta_arr = []
    p_arr = []



    vs.sort(key=lambda x: to_polar(x)[1])
    for i in range(len(vs)):
        theta = math.atan2(vs[(i+1) % len(vs)][1] - vs[i][1], -(vs[i+1% len(vs)][0] - vs[i][0]))
        theta.append(theta)
        
    return vs


def main():
    img = cv2.imread("sample_data/syun.jpg")
    img = enhance(img)
    cv2.imwrite("outputs/enhanced.png", img)
    cons = sub_contour(img)
    for i, con in enumerate(cons):
        edge = edge_detection(con)
        cv2.imwrite(f"outputs/tmp_{i}.png", edge)
        vs = vertex_extract_and_smooth(edge)
        print(vs)
        continue
        canvas = np.zeros_like(edge)
        cv2.polylines(canvas, [vs], True, (255, 0, 0), 2)
        cv2.imwrite(f"outputs/poly_{i}.png", canvas)


if __name__ == "__main__":
    main()
