import argparse
import os
import cv2
import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

import warnings
import numpy as np
import glob
import PIL.Image as Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

warnings.filterwarnings("ignore")
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###  加载jpg文件时使用
def make_dataset(root):
    imgs = []
    # n = len(os.listdir(root)) // 2
    print(root)
    path = root + '/imgs/*.*'
    list = glob.glob(path)
    n = len(list)
    print("test data number:", n)
    crack_path = root + '/imgs/*.*'
    mask_path = root + '/masks/*.*'
    data_nam = glob.glob(crack_path)
    mask_data = glob.glob(mask_path)
    # print(mask_data)
    for i in range(n):
        img = os.path.join(data_nam[i])
        mask = os.path.join(mask_data[i])
        imgs.append((img, mask))
    return imgs


class GetDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('RGB')
        if self.transform is not None:
            img_x = self.transform(img_x)
            # print(img_x.shape)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            # print(img_y.shape)
        return img_x, img_y, x_path

    def __len__(self):
        return len(self.imgs)


## compute
smooth = 0.00001


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def accuracy(prediction, groundtruth):
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float64(np.sum((prediction == 0) & (groundtruth == 1)))
    TN = np.float64(np.sum((prediction == 0) & (groundtruth == 0)))
    accur = float(TP + TN) / (float(TP + FP + FN + TN) + 1e-6)
    return accur


def senstivity(prediction, groundtruth):
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1)))
    FN = np.float64(np.sum((prediction == 0) & (groundtruth == 1)))
    sens = float(TP) / (float(TP + FN) + 1e-6)
    return sens


def iou(prediction, groundtruth):
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float64(np.sum((prediction == 0) & (groundtruth == 1)))
    iou = float(TP) / (float(TP + FN + FP) + 1e-6)
    return iou


def recall(prediction, groundtruth):
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float64(np.sum((prediction == 0) & (groundtruth == 1)))
    rec = float(TP) / (float(TP + FN) + 1e-6)
    prec = float(TP) / (float(TP + FP) + 1e-6)
    F1 = (2 * rec * prec) / (rec + prec + 1e-6)
    return rec, F1


def specificity(prediction, groundtruth):
    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0)))
    TN = np.float64(np.sum((prediction == 0) & (groundtruth == 0)))
    spec = float(TN) / (float(TN + FP) + 1e-6)
    return spec


def Precision(prediction, groundtruth):
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0)))
    Pre = float(TP) / (float(TP + FP) + 1e-6)
    return Pre


def DSC(prediction, groundtruth):
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float64(np.sum((prediction == 0) & (groundtruth == 1)))
    dsc = float(2 * TP) / (float(TP + FP + TP + FN) + 1e-6)
    return dsc


def compute(dataset, title, save_path, model):
    dataset = dataset
    title = title

    path1 = r'./%s' % save_path
    path2 = r'./data/%s/test/masks' % dataset
    list1 = os.listdir(path1)
    list2 = os.listdir(path2)

    L = len(list1)

    Recall_means = 0.
    F1_means = 0.
    piex = 127
    Senstivity_means = 0.
    Accuracy_means = 0.
    Specificity_means = 0.
    Dice_means = 0.
    IOU_means = 0.
    Precision_means = 0.
    for i, j in enumerate(list2):
        img1 = cv2.imread(os.path.join(path1, list1[i]), 0)
        img2 = cv2.imread(os.path.join(path2, list2[i]), 0)
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img11 = np.where(img1 >= piex, 1, 0)
        img22 = np.where(img2 > 0, 1, 0)
        rec_num, f1_num = recall(img11, img22)
        Recall_means += rec_num
        F1_means += f1_num
        dice_num = dice_coef(img11, img22)
        Dice_means += dice_num
        iou_num = iou(img11, img22)
        IOU_means += iou_num
        acc_num = accuracy(img11, img22)
        Accuracy_means += acc_num
        sen_num = senstivity(img11, img22)
        Senstivity_means += sen_num
        spe_num = specificity(img11, img22)
        Specificity_means += spe_num
        pre_num = Precision(img11, img22)
        Precision_means += pre_num
    Sens = Senstivity_means / len(list2)
    Prec = Precision_means / len(list2)
    F1_score = (2 * Prec * Sens) / (Sens + Prec)

    rec = "%s, Senstivity:[%0.6f], Recall:[%0.6f], Specificity:[%0.6f], IOU:[%0.6f], DICE:[%0.6f], Accuracy:[%0.6f], Precision:[%0.6f]," \
          " F1_score:[%0.6f], All_score:[%0.6f]" % (
              title, Senstivity_means / L, Recall_means / L, Specificity_means / L, IOU_means / L, Dice_means / L,
              Accuracy_means / L, Precision_means / L,
              F1_score,
              (Senstivity_means / L + Precision_means / L + F1_means / L)
          )
    g = open('./results/test-image-score-train.txt', 'a')
    g.write(rec + '\n')
    g.close()
    print(rec)


# Dice value
def Dice(inp, target, eps=1):
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)
    # 返回值，让值在0和1之间波动
    return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)


def dice_value(pred, target):
    num = pred.size(0)
    m1 = pred.contiguous().view(num, -1)
    m2 = target.view(num, -1)
    interesection = (m1 * m2).sum()
    return (2. * interesection) / (m1.sum() + m2.sum())


x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()


def test_model(model, dataload, dataset, title, save_load, used_model):
    step = 0
    for x, y, name in dataload:
        step += 1
        inputs = x.to(device)

        outputs = torch.sigmoid(torch.cat([model(inputs)] * 3, dim=1))[0, :, :, :]
        img_y = torch.squeeze(outputs).detach().cpu().numpy()
        img_y = img_y.transpose(1, 2, 0)

        plt.imsave('%s/%s.png' % (save_load, name[0].split("\\")[-1].split(".")[0]), img_y) 

        del inputs
        del outputs
        del img_y

    # 计算指标
    compute(dataset, title, save_load, used_model)


# 测试模型
def test(used_model, data_nam, model):
    batch_size = 1
    test_dataset = GetDataset("data/%s/test" % data_nam, transform=x_transforms, target_transform=y_transforms)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    title = '%s-%s' % (used_model, data_nam)
    save_load = 'results/%s/%s' % (data_nam, used_model)
    os.makedirs(save_load, exist_ok=True)
    model.load_state_dict(
        torch.load(
            'saved_weights/%s-%s/%s_%s_DiceBCE_bestA_model.pth' % (data_nam, used_model, used_model, data_nam),
            map_location='cpu'))
    test_model(model, test_data, data_nam, title, save_load, used_model)



if __name__ == '__main__':

    from models.DCCUNet import DCCUNet

    parse = argparse.ArgumentParser()
    parse.add_argument("--epoch", type=int, default=1, help="the start of epoch")
    parse.add_argument("--num_epoch", type=int, default=100, help="the number of epoches")
    parse.add_argument("--batch_size", type=int, default=1)
    args = parse.parse_args()
    print(args)
    os.makedirs("results/test_record/img_record", exist_ok=True)

    torch.cuda.set_device(0)
    model_list = ["DCCUNet"]
    data_list = ["DSB"]

    model_list = model_list[0:]
    data_list = data_list[0:]

    for j in range(0, len(model_list)):
        for i in range(0, len(data_list)):
            if model_list[j] == 'DCCUNet':
                model = DCCUNet(3, 1).to(device)
                print("--------------Used model: [ %s ]----------------" % model_list[j])
                test(model_list[j], data_list[i], model)

