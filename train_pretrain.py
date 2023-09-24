import time

import torch
import numpy as np
import dataset
import strokenet
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

BATCH_SIZE = 8


def train():
    train_ds = dataset.CalligraphyPretrainDataset("./datasets/sim/train.txt")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_ds = dataset.CalligraphyPretrainDataset("./datasets/sim/test.txt")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = strokenet.StrokeTransformer().to(device)
    net = strokenet.SimpleModel().to(device)
    # net = strokenet.StrokeNet().to(device)
    # criterion = nn.MSELoss()
    def criterion(x, y): return torch.square(x - y)
    lr = 5e-4
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter(
        "./logs/{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    global_step = 0
    best_val_loss = np.inf
    best_train_loss = np.inf
    lr_update_cnt = 0
    early_stop_cnt = 0

    for epoch in range(10000):
        net.train()
        # load data
        train_loss_avg = 0.
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            global_step += 1
            trans, padding, stroke_images, full_image = data
            padding = padding.to(device)
            stroke_images = stroke_images.to(
                device).view(-1, stroke_images.size()[1], 1, 256, 256)
            full_image = full_image.to(device).view(-1, 1, 256, 256)
            trans = trans.to(device)
            # forward
            # forward(self, full_img, stroke_img, output, mask=None):
            # pred_trans = net(full_image, stroke_images, padding[:, :, 0])
            # pred_trans = net(full_image, stroke_images, trans, padding[:, :, 0])
            pred_trans = net(stroke_images)
            loss = torch.masked_select(criterion(pred_trans, trans),
                                       padding).mean()
            train_loss_avg = (train_loss_avg * i + loss.item()) / (i + 1)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            if i % 10 == 0:
                writer.add_scalar("loss", loss.item(), global_step=global_step)
        # validation
        with torch.no_grad():
            net.eval()
            val_loss_mean = 0
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                trans, padding, stroke_images, full_image = data
                padding = padding.to(device)
                stroke_images = stroke_images.to(
                    device).view(-1, stroke_images.size()[1], 1, 256, 256)
                full_image = full_image.to(device).view(-1, 1, 256, 256)
                trans = trans.to(device)
                # pred = net(full_image, stroke_images, padding[:, :, 0])
                # pred = net(full_image, stroke_images, trans, padding[:, :, 0])
                pred = net(stroke_images)
                loss = torch.masked_select(criterion(pred, trans), padding).mean()
                # loss = criterion(pred, trans).mean()
                val_loss_mean += loss.item()
            val_loss_mean /= len(val_loader)
            print(
                f"\nepoch {epoch} train_loss {train_loss_avg:.6f} val_loss {val_loss_mean:.6f}", flush=True)
            writer.add_scalar("val_loss", val_loss_mean,
                              global_step=global_step)
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                early_stop_cnt = 0
                torch.save(net.state_dict(), "./checkpoints/strokenet.pth")
                print("Save model to ./checkpoints/strokenet.pth")
            else:
                early_stop_cnt += 1
                if early_stop_cnt >= 50:
                    print("Early stop, best val loss: {}".format(best_val_loss))
                    break

            if train_loss_avg < best_train_loss:
                best_train_loss = train_loss_avg
                lr_update_cnt = 0
            else:
                lr_update_cnt += 1
                if lr_update_cnt > 5 and lr > 5e-6:
                    lr_update_cnt = 0
                    lr *= 0.5
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                    print("lr updated to {}".format(lr))

            if train_loss_avg < 1 / (7-train_ds.std_level) / 2:
                print("std_level updated to {}".format(train_ds.std_level + .2))
                train_ds.set_std_level(train_ds.std_level + .2)
                val_ds.set_std_level(val_ds.std_level + .2)
                # reset optimizer
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                # reset counter
                best_val_loss = np.inf
                best_train_loss = np.inf
                lr_update_cnt = 0
                early_stop_cnt = 0
                lr = 5e-4
                # reset dataloader
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


if __name__ == '__main__':
    train()
