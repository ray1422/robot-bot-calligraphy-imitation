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
    # ds = dataset.CalligraphyDataset("./datasets/stroke_new")
    ds = dataset.CalligraphyDataset("./datasets/sim", augment=True)
    train_ds, val_ds = ds.split()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = strokenet.StrokeTransformer().to(device)
    # net = strokenet.SimpleModel().to(device)
    net = strokenet.DNN().to(device)
    # criterion = nn.MSELoss()
    def criterion(x, y): return torch.square(x - y)
    lr = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    writer = SummaryWriter(
        "./logs/{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    global_step = 0
    best_val_loss = np.inf
    best_train_loss = np.inf
    lr_update_cnt = 0
    early_stop_cnt = -1000  # FIXME

    for epoch in range(1000):
        net.train()
        # load data
        train_loss_avg = 0.
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            global_step += 1
            trans, padding, stroke_images, full_image = data
            # plot full image and exit
            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use('TkAgg')
            # plt.imshow(full_image[0].numpy())
            # plt.show()
            # exit()
            # plot stroke images and exit
            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use('TkAgg')
            # # just plot the first stroke
            # plt.imshow(stroke_images[0][0].numpy())
            # plt.show()
            # exit()


            padding = padding.to(device)
            stroke_images = stroke_images.to(
                device).view(-1, stroke_images.size()[1], 1, 256, 256)
            full_image = full_image.to(device).view(-1, 1, 256, 256)
            trans = trans.to(device)
            # forward

            # pred_trans = net(full_image, stroke_images, trans, padding[:, :, 0])
            pred_trans = net(full_image, stroke_images)
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
                # pred = net(full_image, stroke_images, trans, padding[:, :, 0])
                pred = net(full_image, stroke_images)
                loss = torch.masked_select(criterion(pred, trans), padding).mean()
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
                if early_stop_cnt >= 100:
                    print("Early stop, best val loss: {}".format(best_val_loss))
                    break

            if train_loss_avg < best_train_loss:
                best_train_loss = train_loss_avg
                lr_update_cnt = 0
            else:
                lr_update_cnt += 1
                if lr_update_cnt > 5:
                    lr_update_cnt = 0
                    lr *= 0.5
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                    print("lr updated to {}".format(lr))


if __name__ == '__main__':
    train()
