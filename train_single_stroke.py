import time
import dataset_stroke
from torch.utils.data import DataLoader
import numpy as np
import strokenet
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

BATCH_SIZE = 8
LR = 1e-4


def train():
    ds = dataset_stroke.StrokeDataset("./datasets/stroke", augment=False)
    train_ds, val_ds = ds.split()
    print(f"train_ds: {len(train_ds)}, val_ds: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = strokenet.StrokeSingleVGG().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    writer = SummaryWriter(
        "./logs/{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    best_val_loss = np.inf
    early_stop_cnt = 0
    for epoch in range(1000):
        train_avg_loss = 0.
        valid_avg_loss = 0.
        for i, (img, full_img, param) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            full_img = full_img.to(device)
            param = param.to(device)
            pred = net(img, full_img)
            loss = torch.square(pred - param).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_avg_loss = (train_avg_loss * i + loss.item()) / (i + 1)
        with torch.no_grad():
            net.eval()
            for i, (img, full_img, param) in enumerate(tqdm(val_loader)):
                img = img.to(device)
                full_img = full_img.to(device)
                param = param.to(device)
                pred = net(img, full_img)
                loss = torch.square(pred - param).mean()
                valid_avg_loss += loss.cpu().numpy()
            valid_avg_loss /= len(val_loader)
        writer.add_scalar("train_loss", train_avg_loss, epoch)
        writer.add_scalar("valid_loss", valid_avg_loss, epoch)


        
        print(f"epoch: {epoch}, train_loss: {train_avg_loss:.6f}, valid_loss: {valid_avg_loss:.6f}")

        if valid_avg_loss < best_val_loss:
            best_val_loss = valid_avg_loss
            early_stop_cnt = 0
            torch.save(net.state_dict(), f"./checkpoints/stroke_single_{epoch}.pth")
            print(f"best model saved at epoch {epoch}")
        else:
            early_stop_cnt += 1
            if early_stop_cnt > 10 and epoch > 20 and valid_avg_loss > train_avg_loss:
                print(f"early stop, best model saved with val_loss: {best_val_loss:.6f}")
                break


if __name__ == "__main__":
    train()
