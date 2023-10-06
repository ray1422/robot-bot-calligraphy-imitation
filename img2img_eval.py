import base64
from glob import glob
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import strokenet
import torch
import matplotlib
import torch.nn as nn

MODEL_PATH = "checkpoints/stroke_single_74.pth"
CHAR_ID = "5a2k"

torch.random.manual_seed(48763)


trans_mean_np = np.array([0.8327963,  0.05671397, 0.04100104, 0.02160976, 0.8883327,  0.0242271])

def iou_torch(x, y, mat):
    mat = mat.reshape((x.shape[0], 2, 3))
    x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
    y = y.reshape((y.shape[0], 1, y.shape[1], y.shape[2]))
    grid = nn.functional.affine_grid(mat, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]),
                                     align_corners=False)
    transformed = nn.functional.grid_sample(x, grid, align_corners=False)
    # only calculate the positive part (e.g. white, 1) parts and ignore the negative part (e.g. black, 0) parts
    th = .5
    y = y > th
    transformed = transformed > th
    intersection = (y & transformed).sum(axis=(1, 2, 3))
    union = (y | transformed).sum(axis=(1, 2, 3))
    return (intersection / union).mean()


def evaluate():
    ious = []
    ious_orig = []
    ious_trans_with_mean = []
    with open("datasets/stroke/val.txt", "r") as f:
        val_list = f.readlines()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = strokenet.StrokeSingleRes18().to(device)
    # net = strokenet.StrokeSingleVGG().to(device)
    # net = strokenet.StrokeSingleAttn().to(device)
    net.load_state_dict(torch.load(MODEL_PATH))
    print("model loaded")
    with torch.no_grad():
        net.eval()
        batch_size = 32
        for i in range(len(val_list) // batch_size):  # drop last
            batch = val_list[i * batch_size: (i + 1) * batch_size]
            if len(batch) == 0:
                break
            imgs = []
            imgs_dst = []
            full_imgs = []
            for line in batch:
                line = line.strip()
                src = np.float32(255 - cv2.imread(line, cv2.IMREAD_GRAYSCALE))
                dst = np.float32(255 - cv2.imread(line.replace("s0_", "s1_"), cv2.IMREAD_GRAYSCALE))
                basedir = os.path.dirname(line)
                # extract basename
                basename = os.path.basename(basedir)
                full_img = np.zeros((256, 256), dtype=np.float32)
                try:
                    # decode basename with base64 to get unicode
                    char_name = base64.urlsafe_b64decode(basename).decode("utf-8")
                    # find "{self.root}/../sim_raw/{char_name}/*/full.png"
                    whole_char_file = glob(f"datasets/sim_raw/{char_name}/char*_stroke/full.png")[0]
                    full_img = 255 - cv2.imread(whole_char_file, cv2.IMREAD_GRAYSCALE)
                except Exception as e:
                    print("Error loading whole char image:", basedir, e)
                # plot here for debug 
                # matplotlib.use('TkAgg')
                # ax = plt.subplot(1, 2, 1)
                # ax.imshow(src)
                # ax = plt.subplot(1, 2, 2)
                # ax.imshow(dst)
                # plt.show()

                imgs.append(src)
                imgs_dst.append(dst)
                full_imgs.append(full_img)
            imgs = np.asarray(imgs)
            imgs_dst = np.asarray(imgs_dst)
            imgs = torch.from_numpy(imgs).to(device).reshape((batch_size, 1, 256, 256)).float() / 255.
            imgs_dst = torch.from_numpy(imgs_dst).to(device).reshape((batch_size, 1, 256, 256)).float() / 255.
            full_imgs = torch.from_numpy(np.asarray(full_imgs)).to(device).reshape((batch_size, 1, 256, 256)).float() / 255.

            # plot here for debug
            # matplotlib.use('TkAgg')
            # ax = plt.subplot(1, 2, 1)
            # ax.imshow(imgs[0, 0].cpu().numpy())
            # ax = plt.subplot(1, 2, 2)
            # ax.imshow(imgs_dst[0, 0].cpu().numpy())
            # plt.show()

            pred = net(imgs, full_imgs)
            # grid = torch.nn.functional.affine_grid(pred.reshape((batch_size, 2, 3)), (batch_size, 1, 256, 256), align_corners=False)
            # new_img = torch.nn.functional.grid_sample(imgs, grid, align_corners=False)
            trans_mean = torch.from_numpy(trans_mean_np).to(device).reshape((1, 2, 3)).float().repeat(batch_size, 1, 1)
            iou_score = iou_torch(imgs.reshape(-1, 256, 256), imgs_dst.reshape(-1, 256, 256), pred)
            iou_trans_with_mean = iou_torch(imgs.reshape(-1, 256, 256), imgs_dst.reshape(-1, 256, 256), trans_mean)
            
            iou_orig = iou(imgs[0, 0].cpu().numpy(), imgs_dst[0, 0].cpu().numpy())
            ious.append(iou_score.item())
            ious_orig.append(iou_orig)
            ious_trans_with_mean.append(iou_trans_with_mean.item())
            print(f"batch: {i}, loss: {iou_score.item()}")
            
    print(f"iou: {np.mean(ious)}")
    print(f"iou_orig: {np.mean(ious_orig)}")
    print(f"iou_trans_with_mean: {np.mean(ious_trans_with_mean)}")


def iou(a, b):
    th = .5
    a = a > th
    b = b > th
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return intersection / union


def vis_eval():
    matplotlib.use('TkAgg')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = strokenet.StrokeSingleRes18().to(device)
    # net = strokenet.StrokeSingleVGG().to(device)
    # net = strokenet.StrokeSingleAttn().to(device)
    net.load_state_dict(torch.load(MODEL_PATH))
    # net = torch.load(MODEL_PATH)
    net.eval()
    with torch.no_grad():

        print("model loaded")
        mean = np.array([1.30195317, -0.11637808, -34.75863473, -0.14586607, 1.22194349, -19.675565])
        std = np.array([0.38823106,  0.22298851, 52.97987973, 0.24098456,  0.35953849, 49.14657975])
        for i in range(6):
            gt = 255 - cv2.imread(f"datasets/stroke/{CHAR_ID}/s1_{i}_full.png", cv2.IMREAD_GRAYSCALE)
            src = 255 - cv2.imread(f"datasets/stroke/{CHAR_ID}/s0_{i}_full.png", cv2.IMREAD_GRAYSCALE)
            full_img = np.zeros((256, 256), dtype=np.float32)
            try:
                # decode basename with base64 to get unicode
                char_name = base64.urlsafe_b64decode(CHAR_ID).decode("utf-8")
                # find "{self.root}/../sim_raw/{char_name}/*/full.png"
                whole_char_file = glob(f"datasets/sim_raw/{char_name}/char*_stroke/full.png")[0]
                full_img = 255 - cv2.imread(whole_char_file, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                print("Error loading whole char image:", CHAR_ID, e)
            trans_gt = np.load(
                f"datasets/stroke/{CHAR_ID}/mat_{i}.npy") if os.path.exists(f"datasets/stroke/{CHAR_ID}/mat_{i}.npy") else None
            x = torch.from_numpy(src).to(device).view(1, 1, 256, 256).float()
            full_img = torch.from_numpy(full_img).to(device).view(1, 1, 256, 256).float()
            theta = net(x / 255., full_img / 255.)
            # let's just use pytorch's affine transform instead of cv2's
            grid = nn.functional.affine_grid(theta.reshape((1, 2, 3)), (1, 1, 256, 256), align_corners=False)
            new_img = nn.functional.grid_sample(x, grid, align_corners=False).cpu().numpy()[0, 0, :, :]

            iou_val = iou(new_img, gt)
            grid_gt = nn.functional.affine_grid(torch.from_numpy(trans_gt).to(device).reshape((1, 2, 3)), (1, 1, 256, 256), align_corners=False).float()
            bs_gt = nn.functional.grid_sample(x, grid_gt, align_corners=False).cpu().numpy()[0, 0, :, :]
            # bs_gt = cv2.warpAffine(
            #     src, trans_gt[: 2, :],
            #     (256, 256)) if trans_gt is not None else np.zeros(
            #     (256, 256),
            #     dtype=np.uint8)
            print(f"iou: {iou_val}")
            fig = plt.figure(figsize=(9, 3))
            ax = fig.add_subplot(1, 3, 1)
            # stack two (256, 256) into (256, 256, 3)
            stacked = np.dstack((np.uint8(bs_gt), np.uint8(gt), np.uint8(src)))
            ax.imshow(stacked)
            ax.title.set_text("searched/gt/original")
            ax = fig.add_subplot(1, 3, 2)
            plt.title(f"VGG Based, iou: {iou_val}")
            # stack two (256, 256) (pred, searched) into (256, 256, 3)
            stacked = np.dstack((np.uint8(new_img), np.uint8(bs_gt), np.zeros((256, 256), dtype=np.uint8)))
            ax.imshow(stacked)

            ax.title.set_text("output/searched")

            # display stacked images of output, gt
            ax = fig.add_subplot(1, 3, 3)
            # stack two (256, 256) into (256, 256, 3)
            stacked = np.dstack((np.uint8(new_img), np.uint8(gt), np.zeros((256, 256), dtype=np.uint8)))
            ax.imshow(stacked)
            ax.title.set_text("output/gt")
            

            fig.savefig(f"outputs/vis_eval_{i}.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    vis_eval()
    evaluate()
