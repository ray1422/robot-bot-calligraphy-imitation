import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import strokenet
from torch.utils.data import DataLoader
import dataset_stroke
from matplotlib import pyplot as plt
import matplotlib
import time
import cv2
import random
import torch
import numpy as np


SEED = 48763
torch.random.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


BATCH_SIZE = 8
LR = 1e-4


def iou(x, y, mat):
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


def gussian_blur(x, ksize=31, sigma=9.):
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.outer(kernel, kernel.transpose())
    kernel = torch.from_numpy(kernel).float().to(x.device)
    kernel = kernel.view(1, 1, ksize, ksize)
    x = nn.functional.conv2d(x, kernel, padding=ksize // 2)
    return x


def transformed_mse(x, y, mat, gt_trans):
    mat = mat.reshape((x.shape[0], 2, 3))
    x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
    y = y.reshape((y.shape[0], 1, y.shape[1], y.shape[2]))
    grid = nn.functional.affine_grid(mat, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]),
                                     align_corners=False)
    transformed = nn.functional.grid_sample(x, grid, align_corners=False)
    # only calculate the positive part (e.g. white, 1) parts and ignore the negative part (e.g. black, 0) parts
    th = .5
    intersection = intersection = y * transformed
    trans_err = torch.mean(torch.square(mat.reshape(-1, 6) - gt_trans.reshape(-1, 6)))
    # union = torch.clip(y + transformed, 0, 1)
    union = y + transformed
    loss = 1 - torch.sum(intersection) / (torch.sum(union) - torch.sum(intersection) + 1e-4)
    area_diff = torch.abs(torch.mean(y) - torch.mean(transformed))
    # apply gaussian blur before calculating mse
    # blurred_transformed = gussian_blur(transformed)
    # blurred_y = gussian_blur(y)
    # mse_loss = torch.sum(torch.square(blurred_y - blurred_transformed) * y) / torch.sum(y) + \
    #     torch.sum(torch.square(blurred_y - blurred_transformed) * (1 - y)) / torch.sum(1 - y)
    # mse_loss = torch.mean(torch.square(blurred_y - blurred_transformed))
    # mse_loss = torch.sum(torch.square(y - transformed) * y) / torch.sum(y) + \
    #     torch.sum(torch.square(y - transformed) * (1 - y)) / torch.sum(1 - y)
    # loss = 1 - ((intersection - union) ** 2).mean()
    # mask = x union y
    # mask = (x > th) | (y > th)
    # loss = ((transformed - y) ** 2 * mask).mean() # / mask.sum()
    # loss = ((transformed - y) ** 2).mean()
    return loss * 4 + trans_err * .5


def train():
    ds = dataset_stroke.StrokePairDataset("./datasets/stroke", augment=False)
    train_ds, val_ds = ds.split()
    print(f"train_ds: {len(train_ds)}, val_ds: {len(val_ds)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = strokenet.StrokeSingleRes18().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    writer = SummaryWriter(
        "./logs/{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    best_val_loss = np.inf
    best_val_iou = 0
    early_stop_cnt = 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1000):

        train_avg_loss = 0.
        valid_avg_loss = 0.

        train_avg_iou = 0.
        valid_avg_iou = 0.

        for i, (img, img_dst, full_img, gt_trans) in enumerate(tqdm(train_loader)):
            img = img.to(device) / 255.
            img_dst = img_dst.to(device) / 255.
            # matplotlib.use('TkAgg')
            # ax = plt.subplot(1, 2, 1)
            # ax.imshow(img[0].cpu().numpy())
            # ax = plt.subplot(1, 2, 2)
            # ax.imshow(img_dst[0].cpu().numpy())
            # plt.show()
            full_img = full_img.to(device) / 255.
            gt_trans = gt_trans.to(device)
            # param = param.to(device)
            pred = net(img, full_img)
            loss = transformed_mse(img, img_dst, pred, gt_trans)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_avg_loss = (train_avg_loss * i + loss.item()) / (i + 1)

            iou_score = iou(img, img_dst, pred)
            train_avg_iou = (train_avg_iou * i + iou_score.item()) / (i + 1)

        with torch.no_grad():
            net.eval()
            for i, (img, img_dst, full_img, gt_trans) in enumerate(tqdm(val_loader)):
                img = img.to(device) / 255.
                img_dst = img_dst.to(device) / 255.
                full_img = full_img.to(device) / 255.
                # param = param.to(device)
                # plot here for debug
                gt_trans = gt_trans.to(device)
                pred = net(img, full_img)
                loss = transformed_mse(img, img_dst, pred, gt_trans)
                valid_avg_loss += loss.cpu().numpy()
                iou_score = iou(img, img_dst, pred)
                valid_avg_iou = (valid_avg_iou * i + iou_score.item()) / (i + 1)
            valid_avg_loss /= len(val_loader)

        writer.add_scalar("train_loss", train_avg_loss, epoch)
        writer.add_scalar("valid_loss", valid_avg_loss, epoch)
        writer.add_scalar("train_iou", train_avg_iou, epoch)
        writer.add_scalar("valid_iou", valid_avg_iou, epoch)

        print(f"epoch: {epoch}, train_loss: {train_avg_loss:.6f}, valid_loss: {valid_avg_loss:.6f}, train_iou: {train_avg_iou:.6f}, valid_iou: {valid_avg_iou:.6f}")

        if valid_avg_iou > best_val_iou:
            best_val_loss = valid_avg_loss
            best_val_iou = valid_avg_iou
            early_stop_cnt = 0
            torch.save(net.state_dict(), f"./checkpoints/stroke_single_{epoch}.pth")
            # torch.save(net.state_dict(),  f"./checkpoints/stroke_single.pth")
            # torch.save(net,  f"./checkpoints/stroke_single.pth")
            print(f"best model saved at epoch {epoch}, IoU: {best_val_iou:.6f}")
        else:
            early_stop_cnt += 1
            if early_stop_cnt > 50 and epoch > 200 and valid_avg_loss > train_avg_loss:
                print(f"early stop, best model saved with val_loss: {best_val_loss:.6f}")
                break


if __name__ == "__main__":
    train()
