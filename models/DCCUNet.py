import torch
import torch.nn as nn
from .base import *


class DCCA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(DCCA, self).__init__()

        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )

        self.CC = CrissCrossAttention(in_channels)
        
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.CC(x)  #1
        x = self.CC(x)  #2
        # x = self.CC(x)  #3
        # x = self.CC(x)  #4
        x = self.pointwise(x)
        return x


class AAM(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(AAM, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1,padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.Softmax(dim=1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input_high, input_low):
        mid_high=self.global_pooling(input_high)
        weight_high=self.conv1(mid_high)

        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        weight=self.conv3(weight_low+weight_high)
        low = self.conv4(input_low)
        high = self.conv4(input_high)
        return high+low.mul(weight)


class DCCUNet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(DCCUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = PACC(img_ch, filters[0])
        self.Conv2 = PACC(filters[0], filters[1])
        self.Conv3 = PACC(filters[1], filters[2])
        self.Conv4 = PACC(filters[2], filters[3])
        self.Conv5 = PACC(filters[3], filters[4])


        self.sk4 = DCCA(filters[3], filters[3],3)
        self.sk3 = DCCA(filters[2], filters[2],3)
        self.sk2 = DCCA(filters[1], filters[1],3)
        self.sk1 = DCCA(filters[0], filters[0],3)

        self.Up5 = UpConv(filters[4], filters[3])
        self.Up_conv5 = DoubleConvBlock(filters[4], filters[3])

        self.Up4 = UpConv(filters[3], filters[2])
        self.Up_conv4 = DoubleConvBlock(filters[3], filters[2])

        self.Up3 = UpConv(filters[2], filters[1])
        self.Up_conv3 = DoubleConvBlock(filters[2], filters[1])

        self.Up2 = UpConv(filters[1], filters[0])
        self.Up_conv2 = DoubleConvBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(
            filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.sk4(e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.sk3(e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.sk2(e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.sk1(e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out

    def name(self):
        return "DCCUNet"


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.con1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.con1(x)
        c2 = self.con2(x)
        c3 = self.con3(x)

        ad = c1/3 + c2/3 + c3/3
        out = self.final(ad)

        return out


class Conv2d_BN(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding, stride=1):
        super(Conv2d_BN, self).__init__()
        self.conv2d_bn = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channels_out),
        )

    def forward(self, x):
        return self.conv2d_bn(x)


class PACC(nn.Module):


    def __init__(self, ch_in, ch_out):
        super(PACC, self).__init__()

        self.con1 = nn.Sequential(
            # Conv2d_BN(ch_in, ch_out, 1, stride=1, padding=0),
            Conv2d_BN(ch_in, ch_out, (1, 3), stride=1, padding=(0, 1)),
            Conv2d_BN(ch_out, ch_out, (3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            Conv2d_BN(ch_out, ch_out, 1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.con2 = nn.Sequential(
            Conv2d_BN(ch_in, ch_out, (1, 5), stride=1, padding=(0, 2)),
            Conv2d_BN(ch_out, ch_out, (5, 1), stride=1, padding=(2, 0)),
            nn.ReLU(inplace=True),
            Conv2d_BN(ch_out, ch_out, 1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.con3 = nn.Sequential(
            # Conv2d_BN(ch_in, ch_out, 1, stride=1, padding=0),
            Conv2d_BN(ch_in, ch_out, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(ch_out, ch_out, (7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(inplace=True),
            Conv2d_BN(ch_out, ch_out, 1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.con1(x)
        c2 = self.con2(x)
        c3 = self.con3(x)

        ad = c1/3 + c2/3 + c3/3
        out = self.final(ad)

        return out


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.rand((1, 3, 256, 256)).to(DEVICE)
    model = DCCUNet().to(DEVICE)
    res = model(image)
    print(res.shape)
