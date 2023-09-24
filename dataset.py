import dataclasses
import os
from typing import List

import torch
import seg
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision
MAX_STROKES_PER_CHAR = 6



@dataclasses.dataclass
class CharData:
    # char_name: str
    transforms: np.ndarray  # (N, 6)
    padding: np.ndarray  # (N, ), bool
    images: np.ndarray  # (N, 256, 256)
    full_image: np.ndarray  # (256, 256)


class CalligraphyPretrainDataset(Dataset):
    def __init__(self, list_file):
        self.data: List[CharData] = []
        self.load_data(list_file)
        self.std_level = 1.8

    def set_std_level(self, std_level):
        if std_level > 6:
            print("std_level too high, set to 6")
            std_level = 6
        self.std_level = std_level

    @staticmethod
    def params2mat(params):
        trans_x, trans_y, angle, scale_x, scale_y, z_bias = params
        # rotation angle in radian, scale in percentage
        m_rotate = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        # translation in pixel
        m_trans = np.asarray([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]], dtype=float)
        m_scale = np.asarray([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=float)
        # m_shear = np.asarray([
        #     [1, shear_x, 0],
        #     [0, 1, 0],
        #     [0, 0, 1]
        # ], dtype=float)
        affine_param = m_trans[:2, :] @ m_scale @ m_rotate  # NOTE: the order is important
        return affine_param

    def load_data(self, list_file):
        base_dir = os.path.dirname(list_file)
        char_names: List[str] = []
        with open(list_file, "r") as f:
            for line in f.readlines():
                char_name = line.strip()
                char_names.append(char_name)

        for char_name in char_names:
            sample = CharData(
                transforms=np.zeros((MAX_STROKES_PER_CHAR, 6), dtype=np.float32),
                padding=np.zeros((MAX_STROKES_PER_CHAR, 6), dtype=np.bool_),
                images=np.zeros((MAX_STROKES_PER_CHAR, 256, 256), dtype=np.float32),
                full_image=np.zeros((256, 256), dtype=np.float32)
            )
            for i in range(MAX_STROKES_PER_CHAR):
                stroke_img_path = os.path.join(base_dir, char_name, f"stroke_{i}.jpg")
                if not os.path.isfile(stroke_img_path):
                    break
                sample.padding[i, :] = True
                stroke_img = \
                    cv2.imdecode(np.fromfile(stroke_img_path,
                                 dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, 0]
                stroke_img = (1 - stroke_img.astype(np.float32) / 255.0) 
                sample.images[i, :, :] = stroke_img
            full_img_path = os.path.join(base_dir, char_name, "full.png")
            # if not os.path.isfile(full_img_path):
            #     full_img_path = os.path.join(base_dir, char_name, "full.png")

            if not os.path.isfile(full_img_path):
                full_img = np.zeros((256, 256), dtype=np.float32)
            else:
                full_img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
                full_img = cv2.resize(full_img, (256, 256))
                full_img = 1 - full_img.astype(np.float32)
            
            sample.full_image = full_img
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # generate transform
        images = self.data[index].images.copy()
        transforms = np.zeros((MAX_STROKES_PER_CHAR, 6), dtype=np.float32)
        for i in range(MAX_STROKES_PER_CHAR):
            if not self.data[index].padding[i, 0]:
                break
            shift_x = np.random.normal(0, 12 / 5 * self.std_level)
            shift_y = np.random.normal(0, 12 / 5 * self.std_level)
            scale_x = np.random.normal(1, 0.36 / 5 * self.std_level)
            scale_y = np.random.normal(1, 0.29 / 5 * self.std_level)
            rotate = np.random.normal(0., .3 / 5 * self.std_level)
            # self.params2mat([shift_x, shift_y, rotate, scale_x, scale_y, 0])[:2, :].reshape(-1)
            transforms[i] = np.asarray([shift_x, shift_y, rotate, scale_x, scale_y, 0], dtype=np.float32)
            # means = np.asarray([[9.3416715e-01,  1.5075799e-02,  6.3704619e+00],
            #                     [-9.9803535e-03,  8.4487855e-01,  1.5373675e+01]]).reshape(-1)

            # stds = np.asarray([[0.3681025,  0.11527948, 49.289494],
            #                    [0.13782544,  0.29989702, 46.81204]]).reshape(-1)
            means = np.asarray([0, 0, 0, 1, 1, 0], dtype=np.float32)
            stds = np.asarray([12, 12, .3, .36, .29, 1.], dtype=np.float32)
            transforms[i] = (transforms[i] - means) / stds
            transform_against = self.params2mat([-shift_x, -shift_y, -rotate, 1/scale_x, 1/scale_y, 0])
            images[i, :, :] = cv2.warpAffine(
                images[i, :, :],
                transform_against,
                (256, 256)
            )

        full_img = self.data[index].full_image.copy()

        return \
            transforms, \
            self.data[index].padding, \
            (images - 0.45) / 0.22, \
            (full_img - 0.45) / 0.22


class CalligraphyDataset(Dataset):
    def __init__(self, dataset_dir, data=None, augment=False):
        super(CalligraphyDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.stroke_img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # noise
            torchvision.transforms.Lambda(lambda x: x + (torch.randn_like(x) - .5) * 0.1),
            # random scale
            # torchvision.transforms.RandomAffine(
            #     degrees=0, translate=(0, 0), scale=(0.8, 1.2), shear=0),
            # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        if data is None:
            self.data: List[CharData] = []
            self.load_data()
        else:
            self.data = data

    @ staticmethod
    def params2mat(params):
        trans_x, trans_y, angle, scale_x, scale_y, z_bias = params
        # rotation angle in radian, scale in percentage
        m_rotate = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        # translation in pixel
        m_trans = np.asarray([[1, 0, trans_x], [0, 1, trans_y]], dtype=float)
        m_scale = np.asarray([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=float)
        # m_shear = np.asarray([
        #     [1, shear_x, 0],
        #     [0, 1, 0],
        #     [0, 0, 1]
        # ], dtype=float)
        affine_param = m_trans @ m_scale @ m_rotate  # NOTE: the order is important
        return affine_param

    def load_data(self):
        # dir structure:
        # <root>/char_name/transform_{i}.npy
        no_full_img_cnt = 0
        for dir_name in os.listdir(self.dataset_dir):
            dir_path = os.path.join(self.dataset_dir, dir_name)
            trans = np.zeros((MAX_STROKES_PER_CHAR, 6), dtype=np.float32)
            padding = np.zeros((MAX_STROKES_PER_CHAR, 6), dtype=np.bool_)
            strokes_img = np.zeros(
                (MAX_STROKES_PER_CHAR, 256, 256), dtype=np.float32)
            # if no .npy file, skip
            if not os.path.isfile(os.path.join(dir_path, "transform.npy")) or \
                    not os.path.isfile(os.path.join(dir_path, "mask.npy")):
                continue

            n_strokes = 0
            trans_data = np.load(os.path.join(dir_path, "transform.npy"))
            mask_data = np.load(os.path.join(dir_path, "mask.npy"))
            # skip if first stroke is not valid
            if not mask_data[0][0]:
                continue
            k = min(np.shape(trans_data)[0], MAX_STROKES_PER_CHAR)
            trans[:k, :] = trans_data[:k, :]
            # padding[:np.shape(mask_data)[0], :] = mask_data
            # ignore if no valid stroke
            if np.sum(mask_data) == 0:
                continue
            for i in range(MAX_STROKES_PER_CHAR):
                file_path = os.path.join(dir_path, f"stroke_{i}.jpg")

                if not os.path.isfile(file_path):
                    break
                # padding[i, :] = True
                # tran = np.load(file_path)
                # # if trans.shape != (7,):
                # #     print(f"invalid transform {dir_name} {i}", tran.shape)
                # #     continue
                if mask_data[i][0]:
                    padding[i, :] = np.asarray([True, True, True, True, True, True])
                else:
                    break
                stroke_img_path = os.path.join(dir_path, f"stroke_{i}.jpg")
                stroke_img = cv2.imdecode(np.fromfile(stroke_img_path,
                                                      dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, 0]
                # # crop to bounding box. background is white and foreground is black
                # _, th = cv2.threshold(stroke_img, 0, 32, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                # # find bounding box
                # # take the largest connected component
                # cnts, _ = cv2.findContours(
                #     th,
                #     cv2.RETR_EXTERNAL,
                #     cv2.CHAIN_APPROX_SIMPLE
                # )
                # cnt = max(cnts, key=cv2.contourArea)
                # x, y, w, h = cv2.boundingRect(cnt)
                # stroke_img = stroke_img[y:y+h, x:x+w]
                # # resize to 256x256
                # stroke_img = cv2.resize(stroke_img, (256, 256))

                # # center padding to 256x256
                # stroke_img = np.pad(stroke_img, ((0, 256 - stroke_img.shape[0]), (0, 256 - stroke_img.shape[1])),
                #                     mode="constant")
                # cast to float32 and normalize
                stroke_img = 1 - stroke_img.astype(np.float32) / 255.0
                # data augmentation

                strokes_img[i, :, :] = stroke_img
                # trans[i, :] = tran
                n_strokes += 1

            # load full image
            # load s0_full.jpg
            full_img_path = os.path.join(dir_path, "full.jpg")
            if not os.path.isfile(full_img_path):
                full_img_path = os.path.join(dir_path, "full.png")

            if not os.path.isfile(full_img_path):
                full_img = np.zeros((256, 256), dtype=np.float32)
                no_full_img_cnt += 1
            else:
                # full_img = \
                #     cv2.imdecode(np.fromfile(full_img_path,
                #                              dtype=np.uint8), cv2.IMREAD_COLOR)
                full_img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
                # full_img = seg.enhance(full_img)[:, :]
                # full_img = full_img[:, :, 0]
                # resize to 256x256
                full_img = cv2.resize(full_img, (256, 256))
                # cast to float32 and normalize
                full_img = 1 - full_img.astype(np.float32) / 255.0
            
            self.data.append(
                CharData(transforms=trans, padding=padding, images=strokes_img, full_image=full_img))

        print(
            f"loaded {len(self.data)} chars, {no_full_img_cnt} chars have no full image")

    def __getitem__(self, index):
        # convert to Tensor
        # (0, height // 4, height // 4 // 3),  # trans_x
        # (0, width // 4, width // 4 // 3),  # trans_y
        # (-0.52, 0.52, 0.34),  # angle from -30 deg to 30 deg but in radian
        # (scale_range[0], scale_range[1], (scale_range[1] - scale_range[0]) / 3),  # scale_x
        # (scale_range[0], scale_range[1], (scale_range[1] - scale_range[0]) / 3),  # scale_y
        # (-0.3, 0.3, 0.2),  # shear_x
        # (-0.3, 0.3, 0.2),  # shear_y

        # calculate means and stds with `trans_norm.py` and hard code here
        # means = [6.370462,   15.373675,   -0.01704985,  0.9429609,   0.857166,   -0.12602071]
        # stds = [49.289494,  46.81204,    0.17302199,  0.36401895, 0.2968497,   0.87630653]
        # use produced transform
        means = np.asarray([[9.3416715e-01,  1.5075799e-02,  6.3704619e+00],
                            [-9.9803535e-03,  8.4487855e-01,  1.5373675e+01]])

        stds = np.asarray([[0.3681025,  0.11527948, 49.289494],
                           [0.13782544,  0.29989702, 46.81204]])

        transform = self.data[index].transforms.copy()
        stroke_img = self.data[index].images.copy()
        full_img = self.data[index].full_image.copy()
        
        if self.augment:
            # augment
            shift_x = np.random.normal(0, 12)
            shift_y = np.random.normal(0, 12)
            scale_x = np.random.normal(1., 0.3)
            scale_y = np.random.normal(1., 0.3)
            rotate = np.random.normal(-0.0, 0.3)
            # trans_x, trans_y, angle, scale_x, scale_y
            # transform[:, 0] += shift_x
            # transform[:, 1] += shift_y
            # transform[:, 3] *= scale_x
            # transform[:, 4] *= scale_y
            # transform[:, 2] += rotate
            # transform images against the transform
            for i in range(MAX_STROKES_PER_CHAR):
                stroke_img[i] = cv2.warpAffine(
                    stroke_img[i],
                    self.params2mat([-shift_x, -shift_y, -rotate, 1/scale_x, 1/scale_y, 0]),
                    (256, 256)
                )
                # add noise
                stroke_img[i] += (np.random.random((256, 256)) - .5) * 0.1
                # stroke_img[i] = self.stroke_img_transform(stroke_img[i])
                # cv2.imshow("stroke", stroke_img[i, :, :] * 255)
                # cv2.waitKey(0)

            full_img = cv2.warpAffine(
                full_img,
                self.params2mat([-shift_x, -shift_y, -rotate, 1./scale_x, 1./scale_y, 0]),
                (256, 256)
            )
            # add noise
            full_img += (np.random.random((256, 256)) - .5) * 0.1
            # cv2.imshow("full", full_img * 255)
            # cv2.waitKey(0)

            # full_img = self.stroke_img_transform(full_img)
        prod = np.zeros((MAX_STROKES_PER_CHAR, 2, 3), dtype=np.float32)
        for i in range(MAX_STROKES_PER_CHAR):
            prod[i, :, :] = self.params2mat(transform[i, :])

        prod = np.reshape(prod, (-1, 6))

        # normalize
        means = means[:2, :].reshape(-1)
        stds = stds[:2, :].reshape(-1)
        normalized_trans = (prod - means) / stds
        # cast to float32
        normalized_trans = np.asarray(normalized_trans, dtype=np.float32)

        # return \
        #     self.data[index].transforms, self.data[index].images, self.data[index].full_image

        return \
            normalized_trans,\
            self.data[index].padding, \
            stroke_img, \
            full_img if np.random.random() > .5 or True \
            else np.array(np.random.random((256, 256)), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def split(self, train_ratio=0.8):
        # set seed
        np.random.seed(48763)
        train_len = int(len(self.data) * train_ratio)
        # split data without reloading
        # shuffle
        np.random.shuffle(self.data)
        train_data = self.data[:train_len]
        valid_data = self.data[train_len:]
        train_dataset = CalligraphyDataset(self.dataset_dir, train_data, augment=self.augment)
        valid_dataset = CalligraphyDataset(self.dataset_dir, valid_data, augment=False)
        return train_dataset, valid_dataset


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    ds = CalligraphyPretrainDataset("./datasets/sim/train.txt")
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    # sample one batch
    trans, padding, stroke_images, full_image = next(iter(loader))
    print(padding[0])
    print(trans.shape, padding.shape, stroke_images.shape, full_image.shape)

    # plot
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    # plot 6 strokes and visualize trans matrix
    for i in range(6):
        plt.subplot(3, 4, i*2+1)
        plt.imshow(stroke_images[0, i, :, :], cmap="gray")
        plt.subplot(3, 4, i*2+2)
        plt.imshow(trans[0, i, :].reshape(2, 3), cmap="gray")
        plt.title(f"stroke {i}")
    plt.show()
    # plot full image
    plt.imshow(full_image[0, :, :], cmap="gray")
    plt.show()

    plt.imshow(padding[0, :, :])
    plt.show()
