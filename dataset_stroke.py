from typing import Optional
import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from glob import glob
import os
import base64
import torchvision
# dataset: (grayscale img, 1x3 param)


class StrokePairDataset(Dataset):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        # random affine transform
        torchvision.transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=5),
        torchvision.transforms.ToTensor(),

    ])

    def __init__(self, root=None, images_a=None, images_b=None, whole_char_images=None, trans=None, augment=False):
        self.root = root
        self.augment = augment
        if root is not None:
            self.files = glob(root + "/*/s0_*_full.png")
            self.trans: Optional[np.ndarray] = None
            self.images_a: Optional[np.ndarray] = None
            self.images_b: Optional[np.ndarray] = None
            self.whole_char_images: Optional[np.ndarray] = None
            self.load_data()
        else:
            self.images_a = images_a
            self.images_b = images_b
            self.whole_char_images = whole_char_images
            self.trans = trans

    def load_data(self):
        kick_list = []
        for file in self.files:
            # check if corresponding stroke transform exists
            # file is s0_{i}_crop.png
            # we need to find the corresponding s1_{i}_crop.png
            corr_img = file.replace("s0_", "s1_")
            mat_file = file.replace("s0_", "mat_").replace("_full.png", ".npy")
            # print(transform_file)
            if not os.path.exists(corr_img) or not os.path.exists(mat_file):
                kick_list.append(file)

        for file in kick_list:
            self.files.remove(file)

        # load images and transforms
        self.images_a = np.zeros((len(self.files), 256, 256), dtype=np.float32)
        self.images_b = np.zeros((len(self.files), 256, 256), dtype=np.float32)
        self.trans = np.zeros((len(self.files), 6), dtype=np.float32)
        self.whole_char_images = np.zeros((len(self.files), 256, 256), dtype=np.float32)
        for i, file in enumerate(self.files):
            self.images_a[i] = 255 - cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            self.images_b[i] = 255 - cv2.imread(file.replace("s0_", "s1_"), cv2.IMREAD_GRAYSCALE)
            self.trans[i] = \
                np.load(file.replace("s0_", "mat_").replace("_full.png", ".npy"))[:2, :].flatten()
            # plot here for debug and exit
            # matplotlib.use('TkAgg')
            # ax = plt.subplot(1, 2, 1)
            # ax.imshow(self.images_a[i])
            # ax = plt.subplot(1, 2, 2)
            # ax.imshow(self.images_b[i])
            # plt.show()
            # load from self.root/../sim-raw/{char_name}/*/full.png
            # find with glob
            # print(whole_char_file)
            # find whole char image and load
            # get basedir name
            basedir = os.path.dirname(file)
            # extract basename
            basename = os.path.basename(basedir)
            try:
                # decode basename with base64 to get unicode
                char_name = base64.urlsafe_b64decode(basename).decode("utf-8")
                # find "{self.root}/../sim_raw/{char_name}/*/full.png"
                whole_char_file = glob(f"{self.root}/../sim_raw/{char_name}/char*_stroke/full.png")[0]
                self.whole_char_images[i] = 255 - cv2.imread(whole_char_file, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                print("Error loading whole char image:", basedir, e)
            # cv2.imshow("test.png", self.whole_char_images[i])
            # cv2.waitKey(0)

    def split(self, ratio=0.8):

        r = int(len(self.files) * ratio)
        # write files list to train.txt and val.txt
        with open(self.root + "/train.txt", "w") as f:
            for file in self.files[:r]:
                f.write(file + "\n")
        with open(self.root + "/val.txt", "w") as f:
            for file in self.files[r:]:
                f.write(file + "\n")
        train_ds = StrokePairDataset(images_a=self.images_a[:r],
                                     images_b=self.images_b[:r],
                                     whole_char_images=self.whole_char_images[:r],
                                     trans=self.trans[:r],
                                     augment=self.augment)
        val_ds = StrokePairDataset(images_a=self.images_a[r:],
                                   images_b=self.images_b[r:],
                                   trans=self.trans[r:],
                                   whole_char_images=self.whole_char_images[r:],
                                   augment=False)

        return train_ds, val_ds

    def __len__(self):
        return len(self.images_a)

    def __getitem__(self, idx):
        # find corresponding stroke transform from file
        # normalize transform by dividing std
        if self.augment:
            return self.transform(
                self.images_a[idx])[
                0, :, :], self.images_b[idx], self.whole_char_images[idx], self.trans[idx]
        return self.images_a[idx], self.images_b[idx], self.whole_char_images[idx], self.trans[idx]


def test_pair_sample():
    ds = StrokePairDataset("./datasets/stroke", augment=True)
    train_ds, val_ds = ds.split()
    print(len(train_ds), len(val_ds))
    print(train_ds[0][0].shape, train_ds[0][1].shape)
    print(val_ds[0][0].shape, val_ds[0][1].shape)

    # show image
    matplotlib.use('TkAgg')
    for img_a, img_b in ds:
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img_a)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img_b)
        plt.show()


def test_sample():
    ds = StrokeDataset("./datasets/stroke", augment=True)
    train_ds, val_ds = ds.split()
    print(len(train_ds), len(val_ds))
    print(train_ds[0][0].shape, train_ds[0][1].shape)
    print(val_ds[0][0].shape, val_ds[0][1].shape)

    # show image
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    for img, full_img,  trans in ds:
        ax = plt.subplot(1, 3, 1)
        ax.imshow(img)
        ax = plt.subplot(1, 3, 2)
        ax.imshow(full_img)
        ax = plt.subplot(1, 3, 3)
        ax.imshow(trans.reshape((2, 3)))
        plt.show()


class StrokeDataset(Dataset):
    def __init__(self, root=None, images=None, whole_char_images=None, transforms=None, augment=False):
        self.root = root
        self.augment = augment
        if root is not None:
            self.files = glob(root + "/*/s0_*_full.png")
            self.images: Optional[np.ndarray] = None
            self.whole_char_images: Optional[np.ndarray] = None
            self.transforms: Optional[np.ndarray] = None
            self.load_data()
        else:
            self.images = images
            self.transforms = transforms
            self.whole_char_images = whole_char_images

    def load_data(self):
        kick_list = []
        for file in self.files:
            # check if corresponding stroke transform exists
            # file is s0_{i}_crop.png
            # we need to find transform_{i}.npy in the same folder
            # transform_file = file.replace("s0_", "transform_").replace("_full.png", ".npy")
            transform_file = file.replace("s0_", "mat_").replace("_full.png", ".npy")
            # print(transform_file)
            if not os.path.exists(transform_file):
                kick_list.append(file)

        for file in kick_list:
            self.files.remove(file)

        # load images and transforms
        self.images = np.zeros((len(self.files), 256, 256), dtype=np.float32)
        self.transforms = np.zeros((len(self.files), 6), dtype=np.float32)
        self.whole_char_images = np.zeros((len(self.files), 256, 256), dtype=np.float32)
        for i, file in enumerate(self.files):
            self.images[i] = 255 - cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            self.transforms[i] = \
                np.load(file.replace("s0_", "mat_").replace("_full.png", ".npy"))[:2, :].flatten()

            # load from self.root/../sim-raw/{char_name}/*/full.png
            # find with glob
            # print(whole_char_file)
            # find whole char image and load
            # get basedir name
            basedir = os.path.dirname(file)
            # extract basename
            basename = os.path.basename(basedir)
            try:

                # decode basename with base64 to get unicode
                char_name = base64.urlsafe_b64decode(basename).decode("utf-8")
                # find "{self.root}/../sim_raw/{char_name}/*/full.png"
                whole_char_file = glob(f"{self.root}/../sim_raw/{char_name}/char*_stroke/full.png")[0]
                self.whole_char_images[i] = 255 - cv2.imread(whole_char_file, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                print("Error loading whole char image:", basedir, e)
            # cv2.imshow("test.png", self.whole_char_images[i])
            # cv2.waitKey(0)

    def split(self, ratio=0.8):

        r = int(len(self.files) * ratio)
        # write files list to train.txt and val.txt
        with open(self.root + "/train.txt", "w") as f:
            for file in self.files[:r]:
                f.write(file + "\n")
        with open(self.root + "/val.txt", "w") as f:
            for file in self.files[r:]:
                f.write(file + "\n")
        train_ds = StrokeDataset(images=self.images[:r],
                                 whole_char_images=self.whole_char_images[:r],
                                 transforms=self.transforms[:r], augment=self.augment)
        val_ds = StrokeDataset(images=self.images[r:],
                               whole_char_images=self.whole_char_images[r:],
                               transforms=self.transforms[r:], augment=False)

        return train_ds, val_ds

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # find corresponding stroke transform from file
        # normalize transform by dividing std
        img = self.images[idx].copy()
        if self.augment:
            x_shift = np.random.normal(0, 32)
            y_shift = np.random.normal(0, 32)
            scale = np.random.normal(1, 0.2)
            rot = np.random.normal(0, 0.2)

            shift_mat = np.array([[1, 0, x_shift],
                                  [0, 1, y_shift],
                                  [0, 0, 1]])
            scale_mat = np.array([[scale, 0, 0],
                                  [0, scale, 0],
                                  [0, 0, 1]])
            rot_mat = np.array([[np.cos(rot), -np.sin(rot), 0],
                                [np.sin(rot), np.cos(rot), 0],
                                [0, 0, 1]])
            mat = shift_mat @ scale_mat @ rot_mat
            # combine with original transform
            # concat original transform with 1x3 mat [0, 0, 1]
            trans = np.concatenate((np.reshape(self.transforms[idx], (2, 3)), np.array([[0, 0, 1]])))
            trans = (mat @ trans)[:2, :].flatten()
            img = cv2.warpAffine(img, mat[:2, :].reshape((2, 3)), (256, 256))
            # plot
            # matplotlib.use('TkAgg')
            # ax = plt.subplot(1, 3, 1)
            # ax.imshow(img)
            # plt.show()

        # mean = np.array([1.30195317, -0.11637808, -34.75863473, -0.14586607, 1.22194349, -19.675565])
        # std = np.array([0.38823106,  0.22298851, 52.97987973, 0.24098456,  0.35953849, 49.14657975])
        mean = np.array([0.8327963,  0.05671397, 0.04100104, 0.02160976, 0.8883327,  0.0242271])
        std = np.array([0.2525839,  0.24113272, 0.13245505, 0.25688055, 0.27227136, 0.15179066])
        # trans = (self.transforms[idx] - mean) / std
        trans = self.transforms[idx]
        return self.images[idx], self.whole_char_images[idx],  trans


def test_sample():
    ds = StrokeDataset("./datasets/stroke", augment=True)
    train_ds, val_ds = ds.split()
    print(len(train_ds), len(val_ds))
    print(train_ds[0][0].shape, train_ds[0][1].shape)
    print(val_ds[0][0].shape, val_ds[0][1].shape)

    # show image
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    for img, full_img,  trans in ds:
        ax = plt.subplot(1, 3, 1)
        ax.imshow(img)
        ax = plt.subplot(1, 3, 2)
        ax.imshow(full_img)
        ax = plt.subplot(1, 3, 3)
        ax.imshow(trans.reshape((2, 3)))
        plt.show()


if __name__ == "__main__":
    # test_sample()
    test_pair_sample()
