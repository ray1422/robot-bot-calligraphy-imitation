import dataclasses
import os
from typing import List
import seg
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset

MAX_STROKES_PER_CHAR = 32


@dataclasses.dataclass
class CharData:
    # char_name: str
    transforms: np.ndarray  # (N, 7)
    padding: np.ndarray  # (N, ), bool
    images: np.ndarray  # (N, 256, 256)
    full_image: np.ndarray  # (256, 256)


class CalligraphyDataset(Dataset):
    def __init__(self, dataset_dir, data=None):
        super(CalligraphyDataset, self).__init__()
        self.dataset_dir = dataset_dir
        if data is None:
            self.data: List[CharData] = []
            self.load_data()
        else:
            self.data = data

    def load_data(self):
        # dir structure:
        # <root>/char_name/transform_{i}.npy
        no_full_img_cnt = 0
        for dir_name in os.listdir(self.dataset_dir):
            dir_path = os.path.join(self.dataset_dir, dir_name)
            trans = np.zeros((MAX_STROKES_PER_CHAR, 7), dtype=np.float32)
            padding = np.zeros((MAX_STROKES_PER_CHAR, 7), dtype=np.bool_)
            strokes_img = np.zeros(
                (MAX_STROKES_PER_CHAR, 256, 256), dtype=np.float32)
            # if no .npy file, skip
            if not os.path.isfile(os.path.join(dir_path, "transform_0.npy")):
                continue
            n_strokes = 0
            for i in range(MAX_STROKES_PER_CHAR):
                file_path = os.path.join(dir_path, f"transform_{i}.npy")
                if not os.path.isfile(file_path):
                    break
                padding[i, :] = True
                tran = np.load(file_path)
                # if trans.shape != (7,):
                #     print(f"invalid transform {dir_name} {i}", tran.shape)
                #     continue
                stroke_img_path = os.path.join(dir_path, f"s0_{i}.png")
                stroke_img = \
                    cv2.imdecode(np.fromfile(stroke_img_path,
                                 dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, 0]
                # center padding to 256x256
                stroke_img = np.pad(stroke_img, ((0, 256 - stroke_img.shape[0]), (0, 256 - stroke_img.shape[1])),
                                    mode="constant")
                # cast to float32 and normalize
                stroke_img = stroke_img.astype(np.float32) / 255.0
                strokes_img[i, :, :] = stroke_img
                trans[i, :] = tran
                n_strokes += 1

            # load full image
            # load s0_full.jpg
            full_img_path = os.path.join(dir_path, "s0_full.jpg")
            if not os.path.isfile(full_img_path):
                full_img_path = os.path.join(dir_path, "s0_full.png")

            if not os.path.isfile(full_img_path):
                full_img = np.zeros((256, 256), dtype=np.float32)
                no_full_img_cnt += 1
            else:
                # full_img = \
                #     cv2.imdecode(np.fromfile(full_img_path,
                #                              dtype=np.uint8), cv2.IMREAD_COLOR)
                full_img = cv2.imread(full_img_path)
                full_img = seg.enhance(full_img)[:, :]
                # full_img = full_img[:, :, 0]
                # resize to 256x256
                full_img = cv2.resize(full_img, (256, 256))
                # cast to float32 and normalize
                full_img = full_img.astype(np.float32) / 255.0

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
        means = [-2.59199314, -0.87927684, -0.01034304,  1.11926225,  1.12914837, -0.11321369,
                 -0.10528584]
        stds = [11.1822519,   7.88766075, 0.22998645, 0.26682949, 0.3191827, 0.22838493,
                0.26623996]
        normalized_trans = \
            (self.data[index].transforms - means) / stds
        # cast to float32
        normalized_trans = np.asarray(normalized_trans, dtype=np.float32)
        # return \
        #     self.data[index].transforms, self.data[index].images, self.data[index].full_image
        return \
            normalized_trans,\
            self.data[index].padding, \
            self.data[index].images, \
            self.data[index].full_image if np.random.random() > .5 or True\
            else np.zeros((256, 256), dtype=np.float32)

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
        train_dataset = CalligraphyDataset(self.dataset_dir, train_data)
        valid_dataset = CalligraphyDataset(self.dataset_dir, valid_data)
        return train_dataset, valid_dataset
