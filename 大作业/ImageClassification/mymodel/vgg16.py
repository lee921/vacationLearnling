import torch.nn as tnn


# 卷积 + batchNorm +Relu
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


# 四层作为一个block，每个block做一次MaxPool
def vgg_conv_block(in_list, out_list, k_list, p_list):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    return tnn.Sequential(*layers)


# 全连接层
def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, inSize, n_classes=1000):
        super(VGG16, self).__init__()

        # 输入图片32*32*3 , inSize=32
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1])
        # self.maxPooling1=tnn.MaxPool2d(kernel_size=2, stride=2)
        # 32*32*64
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1])
        self.maxPooling2 = tnn.MaxPool2d(kernel_size=2, stride=2)
        # 16*16*128 图像尺寸/2
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1])
        self.maxPooling3 = tnn.MaxPool2d(kernel_size=2, stride=2)
        # 8*8*256 图像尺寸/2
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1])
        self.maxPooling4 = tnn.MaxPool2d(kernel_size=2, stride=2)
        # 4*4*512 图像尺寸/2
        self.layer5 = vgg_conv_block([512, 1024, 1024], [1024, 1024, 1024], [3, 3, 3], [1, 1, 1])
        self.maxPooling5 = tnn.MaxPool2d(kernel_size=2, stride=2)
        # 2*2*1024   图像尺寸/2
        # FC layers
        self.layer6 = vgg_fc_layer(int(inSize / 16) * int(inSize / 16) * 1024, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)

        out = self.layer2(out)
        out = self.maxPooling2(out)

        out = self.layer3(out)
        out = self.maxPooling3(out)

        out = self.layer4(out)
        out = self.maxPooling4(out)

        out = self.layer5(out)
        out = self.maxPooling5(out)
        vgg16_features = out
        out = out.view(out.size(0), -1)

        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out
