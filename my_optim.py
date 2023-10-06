import glob
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch.autograd import Variable
import tqdm
import torch.nn as nn 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gussian_blur(x, ksize=31, sigma=9.):
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.outer(kernel, kernel.transpose())
    kernel = torch.from_numpy(kernel).float().to(x.device)
    kernel = kernel.view(1, 1, ksize, ksize)
    x = nn.functional.conv2d(x, kernel, padding=ksize // 2)
    return x


def search(img_a, img_b, prog_bar=None):
    mat = Variable(torch.Tensor([
        [1, 0, 0],
        [0, 1, 0]
    ]).to(device), requires_grad=True)

    optimizer = torch.optim.Adam([mat], lr=1e-2)

    best_loss = 1e10
    lr_cnt = 0
    lr = 1e-1
    ious = [999, 999, 999]
    best_iou = 0
    ret = None
    for step in range(1000):
        img_a_t = torch.from_numpy(img_a).float().reshape((1, 1, 256, 256)).to(device)
        img_b_t = torch.from_numpy(img_b).float().reshape((1, 1, 256, 256)).to(device)
        grid = torch.nn.functional.affine_grid(mat.reshape((1, 2, 3)), (1, 1, 256, 256), align_corners=False)
        new_img = torch.nn.functional.grid_sample(img_a_t, grid, align_corners=False)

        new_img_blur = gussian_blur(new_img)
        img_b_t_blur = gussian_blur(img_b_t)
        # plot here for debug
        # matplotlib.use('TkAgg')
        # ax = plt.subplot(1, 2, 1)
        # ax.imshow(new_img_blur[0, 0].detach().cpu().numpy())
        # ax = plt.subplot(1, 2, 2)
        # ax.imshow(img_b_t_blur[0, 0].detach().cpu().numpy())
        # plt.show()

        # loss = torch.mean(torch.square(new_img - img_b_t))
        iou_loss = 1 - torch.sum(img_b_t * new_img) / (torch.sum(img_b_t + new_img) - torch.sum(img_b_t * new_img) + 1e-6)
        blurred_mse = torch.mean(torch.square(new_img_blur - img_b_t_blur))
        loss = iou_loss + blurred_mse
        th = .5
        new_img = new_img > th
        img_b_t = img_b_t > th
        intersection = (new_img & img_b_t).sum()
        union = (new_img | img_b_t).sum()
        iou = intersection / union
        # if step % 50 == 0:
        #     print(f"step: {step}, loss: {loss.item():.4f}, iou: {iou.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ious[step % 3] = iou.item()
        if iou.item() > best_iou:
            best_iou = iou.item()
            best_loss = loss.item()
            ret = mat.detach().cpu().clone()
        elif abs(ious[0] - ious[1]) < 1e-6 and abs(ious[1] - ious[2]) < 1e-6 and best_iou > .8:
            if prog_bar is None:
                print(f"converged at step {step}, loss: {best_loss:.4f}, iou: {best_iou:.4f}")
            else:
                prog_bar.set_description(f"converged at step {step}, loss: {best_loss:.4f}, iou: {best_iou:.4f}")
            break
        else:
            lr_cnt += 1
            if lr_cnt > 50:
                lr_cnt = 0
                lr /= 2
                optimizer = torch.optim.Adam([mat], lr=lr)
    return ret if best_iou > .5 else None


def main():
    chars = [x for x in os.listdir("datasets/stroke/") if os.path.isdir(os.path.join("datasets/stroke/", x))]
    prog_bar = tqdm.tqdm(chars)
    for char in prog_bar:
        files = glob.glob(os.path.join("datasets/stroke/", char, "s0_*_full.png"))
        for file in files:
            img_a = np.float32(255 - cv2.imread(file, cv2.IMREAD_GRAYSCALE)) / 255
            img_b = np.float32(255 - cv2.imread(file.replace("s0_", "s1_"), cv2.IMREAD_GRAYSCALE)) / 255
            mat = search(img_a, img_b, prog_bar)
            if mat is None:
                print(f"failed to converge on {file}")
                continue
            grid = torch.nn.functional.affine_grid(mat.reshape((1, 2, 3)), (1, 1, 256, 256), align_corners=False)
            new_img = torch.nn.functional.grid_sample(torch.from_numpy(img_a).float().reshape((1, 1, 256, 256)),
                                                      grid, align_corners=False)
            # save the mat
            np.save(file.replace("s0_", "mat_").replace("_full.png", ".npy"), mat.detach().numpy())

    img_a = np.float32(255 - cv2.imread("datasets/stroke/5a-6/s0_2_full.png", cv2.IMREAD_GRAYSCALE)) / 255
    img_b = np.float32(255 - cv2.imread("datasets/stroke/5a-6/s1_2_full.png", cv2.IMREAD_GRAYSCALE)) / 255

    mat = search(img_a, img_b)
    grid = torch.nn.functional.affine_grid(mat.reshape((1, 2, 3)), (1, 1, 256, 256), align_corners=False)
    new_img = torch.nn.functional.grid_sample(torch.from_numpy(img_a).float().reshape((1, 1, 256, 256)),
                                              grid, align_corners=False)
    # plot the result
    matplotlib.use('TkAgg')
    ax = plt.subplot(1, 4, 1)
    ax.imshow(img_a)
    ax = plt.subplot(1, 4, 2)
    ax.imshow(img_b)
    ax = plt.subplot(1, 4, 3)
    ax.imshow(new_img[0, 0].detach().numpy())
    # plot a, b, new together in RGB
    ax = plt.subplot(1, 4, 4)
    ax.imshow(np.stack([np.zeros((256, 256)), img_b, new_img[0, 0].detach().numpy()], axis=2))

    plt.show()


if __name__ == '__main__':
    main()
