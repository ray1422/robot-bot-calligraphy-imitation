"""
robostic caligraphy simulator environment
the weights of strokes are calculated by the distance between the brush and the paper.

Sim class is the simulator. it take trace points as input and output the image of the trace. 
Boundary should be set at initialization.

tracing points should be in the following format:
    movl 0 24.4244 379.5504 183.1498 163.0036 -3.8346 -143.2302 100.0000 stroke1
where row[2:4] is x, y, z, and row[5:7] is the rotation angle of the brush.
the rest of the row is not used in our work.

This file is highly inspired by vis_2d in get_6d_3d.py and somewhere in the project "Robot-Calligraphy".
They're legacy code and I don't know who wrote them. They're unreadable, non-reusable and untraceable, and 
meant to be fully rewritten. As the result, I rewrote them in this file.
The unreadability of the legacy code is not just caused by lack of ability of programming of the author,
but also the English level of the author. The variable names are confusing, misleading and sometimes 
completely wrong. Also, they don't follow the naming convention of python.
"""

import math
from typing import List
import numpy as np
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

from scipy import ndimage
import cv2

# interface


class CalSim(ABC):
    def __init__(self) -> None:
        self.trace_path = None

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """
        get the output image of the trace where strokes are in black and background is in white (255).
        """
        pass


class CalSimSimple(CalSim):
    """
    CalSimSimple is a simple simulator that only consider the distance between 
    the brush and the paper.
    :param trace: the trace of the brush. it should be a 2d numpy array with shape (n, 2).
    :param file: the path of the trace file. it should be a txt file with the following format:
        movl 0 24.4244 379.5504 183.1498 163.0036 -3.8346 -143.2302 100.0000 stroke1
    where angles are in degree.

    """

    def __init__(self, trace=None, trace_3d=None, file=None, boundary=None) -> None:
        super().__init__()
        self.boundary = boundary
        self.trace_3d = None
        self.trace = None

        if trace_3d is not None:
            self.trace_3d = trace_3d
        elif trace is not None:
            self.trace = trace
        else:
            self.trace = self.load_trace_from_file(file)

        if self.trace_3d is None:
            self.trace_3d = list(map(self.trans_6d_to_3d, self.trace))

        if self.boundary is None:
            self.boundary = [0, 0, 10, 10]
            self.boundary[0] = min([x[0] for x in self.trace_3d if x[2] < 10])
            self.boundary[1] = max([x[0] for x in self.trace_3d if x[2] < 10])
            self.boundary[2] = min([x[1] for x in self.trace_3d if x[2] < 10])
            self.boundary[3] = max([x[1] for x in self.trace_3d if x[2] < 10])

    def load_trace_from_file(self, file_path) -> np.ndarray:
        """
        load trace from file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            trace = []
            for line in lines:
                line = line.lstrip().split(' ')
                x, y, z, theta, phi, psi = map(float, line[2:8])
                trace.append([x, y, z, theta, phi, psi])
            return trace

    def split_strokes(self) -> List["CalSimTrans3D"]:
        # split the trace into strokes if height >= 11
        start_idx = 0
        ret = []
        for i, (x, y, z) in enumerate(self.trace_3d):
            # print("z", z)
            if z >= 11:
                if i - start_idx <= 1:
                    start_idx = i+1
                    continue
                ret.append(CalSimTrans3D(trace=self.trace[start_idx:i-1].copy(), boundary=self.boundary.copy()))
                start_idx = i+1
        return ret

    @staticmethod
    def trans_6d_to_3d(point):
        x, y, z, theta, phi, psi = point
        stroke_n = x, y, z, theta, phi, psi
        theta, phi, psi = map(np.deg2rad, [theta, phi, psi])
        a, b, c = theta, phi, psi
        r_a = [1, 0, 0,
               0, math.cos(a), -1 * math.sin(a),
               0, math.sin(a), math.cos(a)]

        r_b = [math.cos(b), 0, math.sin(b),
               0, 1, 0,
               -1 * math.sin(b), 0, math.cos(b)]

        r_c = [math.cos(c), -1 * math.sin(c), 0,
               math.sin(c), math.cos(c), 0,
               0, 0, 1]

        r_a = np.array(r_a).reshape(3, 3)
        r_b = np.array(r_b).reshape(3, 3)
        r_c = np.array(r_c).reshape(3, 3)

        r = np.dot(np.dot(r_c, r_b), r_a)

        a = [r[0, 0], r[0, 1], r[0, 2], x,
             r[1, 0], r[1, 1], r[1, 2], y,
             r[2, 0], r[2, 1], r[2, 2], z,
             0, 0, 0, 1]
        a = np.array(a).reshape((4, 4))

        b = np.identity(4)
        b[2, 3] = 185  # 毛筆長度 185 mm

        t = np.dot(a, b)

        return t[0, 3], t[1, 3], t[2, 3]

    def get_image(self) -> np.ndarray:
        # total width and height is 256
        # scale to 224X224 and the rest is padding
        scale = min(224 / (self.boundary[1] - self.boundary[0]),
                    224 / (self.boundary[3] - self.boundary[2]))

        def trans_x(x): return int((x - self.boundary[0]) * scale) + int(
            256 - (self.boundary[1] - self.boundary[0]) * scale)//2
        def trans_y(y): return 256 - int((y - self.boundary[2]) * scale) - int(
            256 - (self.boundary[3] - self.boundary[2]) * scale)//2
        canvas = np.ones((256, 256), dtype=np.uint8) * 255

        for i in range(0, (len(self.trace_3d) - 1)):
            # print(self.trace_3d[i])
            x = [trans_x(self.trace_3d[i][0]),
                 trans_x(self.trace_3d[i + 1][0])]
            y = [trans_y(self.trace_3d[i][1]),
                 trans_y(self.trace_3d[i + 1][1])]
            h = (float(self.trace_3d[i][2]) +
                 float(self.trace_3d[i + 1][2])) * 0.5

            width = 2 * (5.5 - h)
            if width < 1:
                continue
            # width = 1 # TODO
            # plt.plot(x, y, 'k', color="c", linewidth=width)
            # draw line with OpenCV
            # print(x, y, width)
            cv2.line(canvas, (x[0], y[0]), (x[1], y[1]), (0, 0, 0), int(width))
            # print(x, y, width)
        # cv2.imwrite("./test.png", canvas)
        # print(h)
        return canvas


class CalSimTrans3D(CalSimSimple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_cal_sim_simple(sim: CalSimSimple) -> "CalSimTrans3D":
        """
        convert CalSimSimple to CalSimTrans3D
        """
        return CalSimTrans3D(trace=sim.trace.copy(), boundary=sim.boundary.copy())

    def transform(self, params) -> "CalSimTrans3D":
        """
        apply transformation to the trace and return a new CalSimTrans3D object
        params are (affine on x-z plane, bias on y)
        """
        # parameters for affine transformation
        trans_x, trans_y, angle, scale_x, scale_y, shear_x, shear_y, z_bias = params

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
        new_trace = []
        for i, (x, y, z) in enumerate(self.trace_3d):
            x, y = np.matmul(affine_param, np.array([x, y, 1]))
            new_trace.append([x, y, z + z_bias])

        return CalSimTrans3D(trace_3d=new_trace, boundary=self.boundary.copy())


if __name__ == "__main__":
    # test
    trace_file = "./char00900_stroke.txt"
    sim = CalSimSimple(file=trace_file)
    sim.get_image()
    print("done")

# the following are from legacy code. don't touch it and even use it.
# I just copied them here and referenced them.


class _LegacySim:
    def __init__(self) -> None:
        self.canvas = 255 - np.zeros((256, 256), dtype=np.uint8)
        self.six_d_points = []
        self.points = []

    def paint_from_file(self, file_path) -> None:
        with open(file_path, "r") as f:
            lines = f.readlines()
            x, y, z, theta, phi, psi = 0., 0., 0., 0., 0., 0.
            try:
                x, y, z, theta, phi, psi = lines[0].split(" ")[2:8]
                self.six_d_points.append([x, y, z, theta, phi, psi])
            except ValueError:
                print("invalid line of file: ", file_path)

        # find 3d data from 6-axis data
        for x, y, z, theta, phi, psi in self.six_d_points:
            pass


def legacy_get_6d_3d(path):
    """
    input: 6 axis txt file 
    (movl 0 num1 num2 num3 num4 num5 num6 100.0000 stroke1)

    output: (x, y, z, stroke) visualized data 

    The following is copied from legacy code. don't touch it and even use it.
    """
    data = []
    i = 2
    with open(path) as txtFile:
        for row in txtFile:

            row = row.lstrip().split(',')  # for csv file
            # row = row.lstrip().split(' ')  # for txt file
            x = float(row[0 + i])
            y = float(row[1 + i])
            z = float(row[2 + i])
            a = angle2deg(float(row[3 + i]))
            b = angle2deg(float(row[4 + i]))
            c = angle2deg(float(row[5 + i]))
            stroke_n = row[-1]

            Ra = [1, 0, 0,
                  0, math.cos(a), -1 * math.sin(a),
                  0, math.sin(a), math.cos(a)]

            Rb = [math.cos(b), 0, math.sin(b),
                  0, 1, 0,
                  -1 * math.sin(b), 0, math.cos(b)]

            Rc = [math.cos(c), -1 * math.sin(c), 0,
                  math.sin(c), math.cos(c), 0,
                  0, 0, 1]

            Ra = np.array(Ra).reshape(3, 3)
            Rb = np.array(Rb).reshape(3, 3)
            Rc = np.array(Rc).reshape(3, 3)

            R = np.dot(np.dot(Rc, Rb), Ra)

            A = [R[0, 0], R[0, 1], R[0, 2], x,
                 R[1, 0], R[1, 1], R[1, 2], y,
                 R[2, 0], R[2, 1], R[2, 2], z,
                 0, 0, 0, 1]
            A = np.array(A).reshape((4, 4))

            B = np.identity(4)
            B[2, 3] = 185  # 毛筆長度 185 mm

            T = np.dot(A, B)

            data.append([T[0, 3], T[1, 3], T[2, 3], stroke_n])

    return data


class Vi_c():
    '''
    this class is legacy code. don't touch it and even use it.
    ***For visualizing component

    read_txt: read 3axis txt file to list
    plot: plot from list to image
    savefig: save plot image
    '''

    def __init__(self, d_path):
        self.d_path = d_path
        self.points = []
        self.name = ''
        self.fig = None
        self.ax = None
        self.simulator = False  # 是否顯示字體寬度

    def read_txt(self, f_name):
        self.name = f_name[:-4]
        f = []
        with open(f_name, 'r') as txtFile:
            rows = txtFile.readlines()
            for row in rows:
                row = row.rstrip('\n').split(' ')  # for txt
                # row = row.rstrip('\n').split(',') # for csv
                row[-1] = row[-1].strip('stroke')
                f.append(row)
        self.points = f

    def plot(self, c_name):
        self.fig = plt.figure()

        self.ax = plt.axes()
        self.ax.set_xlabel('X', fontdict={'size': 15, 'color': 'k'})
        self.ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'k'})
        self.ax.set_title('%s' % c_name, fontproperties='SimSun', fontsize=20)
        # stroke = 1
        a = np.arange(0, 1, 0.001)
        c = np.random.choice(a, 3, replace=False)

        for i in range(0, (len(self.points) - 1)):
            x = [float(self.points[i][0]), float(self.points[i + 1][0])]
            y = [float(self.points[i][1]), float(self.points[i + 1][1])]
            h = (float(self.points[i][2]) + float(self.points[i + 1][2])) * 0.5
            if h > 10:
                continue
            # if int(self.points[i][3]) == stroke:
            #     c = np.random.choice(a, 3, replace=False)
            #     self.ax.text(float(self.points[i][0]), float(self.points[i][1]), str(stroke), fontsize=12)
            #     stroke += 1
            else:
                if self.simulator:
                    width = 2 * (5.5 - h)
                    self.ax.plot(x, y, 'k', color=c, linewidth=width)
                else:
                    self.ax.plot(x, y, 'k', color=c)

                # plts.append(self.ax.plot(x, y, 'k', color=c, linewidth=width, label='%d' % stroke)[0]) # 會出現相同lebel的bug.
                # self.ax.plot(x, y, 'k', color=c, linewidth=width)
                # plt.pause(0.05) # 用來顯示一個字的點順序
        # plt.legend(handles=plts, bbox_to_anchor=(1.1, 1))

    def show(self):
        plt.show()
        pass

    def savefig(self, d_save):
        name = self.name.split('/')[-1].split('\\')[-1]
        p_save = os.path.join(d_save, '%s.png' % name)
        # plt.savefig('./data/vis_2d/%s.jpg' % self.name)

        plt.savefig(p_save)
        plt.close()
        pass
