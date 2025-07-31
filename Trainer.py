import argparse
import os
import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

import time
import numpy as np
import glob
import warnings

warnings.filterwarnings("ignore")
import PIL.Image as Image
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataset(root, mode):
    imgs = []
    print("Dataset path", root)
    path = root + '/imgs/*.*'
    list = glob.glob(path)
    n = len(list)
    print("%s image number:" % mode, n)
    crack_root1 = root + '/imgs/*.*'
    mask_root1 = root + '/masks/*.*'
    crack_data = glob.glob(crack_root1)
    crack_label = glob.glob(mask_root1)

    print(len(crack_data), len(crack_label))
    for i in range(len(crack_data)):
        img = os.path.join(crack_data[i])
        mask = os.path.join(crack_label[i])
        imgs.append((img, mask))
    return imgs


class SetDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, mode="Train"):
        imgs = make_dataset(root, mode)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('RGB')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        # print(img_y.shape)
        return img_x, img_y[0,:,:]

    def __len__(self):
        return len(self.imgs)


def Dice(inp, target, eps=1):
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)
    # 返回值，让值在0和1之间波动
    return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)

### MoNuSeg
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.Compose([
    transforms.ToTensor()
])

def dice_value(pred, target):
    num = pred.size(0)
    m1 = pred.contiguous().view(num, -1)
    m2 = target.view(num, -1)
    interesection = (m1 * m2).sum()
    return (2. * interesection) / (m1.sum() + m2.sum())


def dice_loss(pred, target):
    smooth = 1
    num = pred.size(0)
    m1 = pred.contiguous().view(num, -1)
    m2 = target.view(num, -1)
    interesection = (m1 * m2).sum()
    return 1 - ((2. * interesection + smooth) / (m1.sum() + m2.sum() + smooth))


def train_model(epoch, model, criterion, optimizer, dataload, num_epochs, used_model, datasets, save_model):
    save_index  = 6.0
    for epoch in range(epoch, num_epochs):
        prev_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs))
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        dice_loss_all = 0.0
        dice_all = 0.0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            if inputs.size(0) == 1:
                inputs = torch.cat([inputs] * 2, dim=0)
                labels = torch.cat([labels] * 2, dim=0)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs))
            dice = dice_value(outputs, labels)
            dice_all += dice.item()
            dice_loss1 = dice_loss(outputs, labels)
            BCE_loss = criterion(outputs, labels)
            loss = dice_loss1 + BCE_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            dice_loss_all += dice_loss1.item()
            print("Epoch:[%d], %d/%d, %s-%s, train_loss:%0.6f, dice_loss:%0.6f, dice:%0.6f, bce:%0.6f" % (
                epoch, step, (dt_size - 1) // dataload.batch_size + 1, used_model, datasets, loss.item(), dice_loss1,
                dice, BCE_loss.item()))
        epoch_time = time.time()
        print(
            "epoch:%d, used_model:%s, used_data: [%s], train_loss:%0.6f, avg_dice_loss:%0.6f, dice:%0.6f, epoch_time:%0.4f" % (
                epoch, used_model, datasets, epoch_loss / step, dice_loss_all / step, dice_all / step,
                (epoch_time - prev_time)))
        if (epoch_loss / step) <= save_index:
            save_index = epoch_loss / step
            torch.save(model.state_dict(), 'saved_weights/%s/%s_%s_DiceBCE_bestA_model.pth' % (save_model, used_model, datasets))
            print("Save the best model: [%s-%s] epoch, Loss: [%s]" % (used_model, epoch, save_index))


# 训练模型
def train(args, used_model, data_nam, model, save_model):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    liver_dataset = SetDataset("./data/%s/train" % data_nam, transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_model(args.epoch, model, criterion, optimizer, dataloaders, args.num_epoch, used_model, data_nam, save_model)


if __name__ == '__main__':

    from models.DCCUNet import DCCUNet

    print("torch.cuda GPU:", torch.cuda.is_available())
    torch.cuda.set_device(0)

    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("--epoch", type=int, default=1, help="the start of epoch")
    parse.add_argument("--num_epoch", type=int, default=3, help="the number of epoches")
    parse.add_argument("--batch_size", type=int, default=8)
    args = parse.parse_args()
    print(args)

    model_list = ["DCCUNet"]
    data_list = ["DSB"]

    # model_list = model_list[0:3]
    data_list = data_list[0:]

    for i in range(0, len(data_list)):
        for j in range(0, len(model_list)):
            if model_list[j] == 'DCCUNet':
                model = DCCUNet(3, 1).to(device)
                save_model = '%s-%s' % (data_list[i], model_list[j])
                os.makedirs("saved_weights/%s" % save_model, exist_ok=True)
                print("--------------Used model: [ %s ]----------------" % model_list[j])  # # PAC_DW
                train(args, model_list[j], data_list[i], model, save_model)
