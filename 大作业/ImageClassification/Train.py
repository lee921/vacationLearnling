import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from mymodel import vgg16 as VGG
from mymodel import resnet18 as ResNet

BATCH_SIZE = 10
LEARNING_RATE = 0.001
EPOCH = 20
N_CLASSES = 10

# modelname in {vgg16,resnet18}
modelname = 'vgg16'
# 随机裁减图片尺寸，必须是32的整数倍   vgg16:224,resnet18:32
inSize = 32
is_trained = False
pretrainpath = './SaveModelParams/'+modelname+'.pkl'
dssetspath = 'E:\PythonProject\datasets\CIFAR-10'


def loadDataset(dssetspath):
    # load train datasets and test datasets
    transform = transforms.Compose([
        transforms.RandomResizedCrop(inSize),  # 随机裁剪224×224
        transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
        transforms.ToTensor(),  # 将给定图像转为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化处理   均值和方差需要抽样获取
                             std=[0.229, 0.224, 0.225]),
    ])

    # trainData = dsets.ImageFolder('../data/imagenet/train', transform)
    # testData = dsets.ImageFolder('../data/imagenet/test', transform)
    trainData = dsets.CIFAR10(root=dssetspath, train=True, download=True, transform=transform)
    testData = dsets.CIFAR10(root=dssetspath, train=False, download=True, transform=transform)
    return trainData, testData


def getModel(modelname='vgg16', is_pretrained=False):
    model = None
    if modelname == 'vgg16':
        model = VGG.VGG16(inSize=inSize, n_classes=N_CLASSES)
    if modelname == 'resnet18':
        model = ResNet.ResNet(inSize=inSize,num_classes=N_CLASSES)

    model.cuda()
    # is_trained
    if is_pretrained:
        model.load_state_dict(torch.load(pretrainpath))

    return model


if __name__ == '__main__':

    trainData, testData = loadDataset(dssetspath)

    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=8, shuffle=False)

    # get model
    model = getModel(modelname=modelname, is_pretrained=is_trained)

    # Loss, Optimizer & Scheduler
    cost = tnn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Train the model
    for epoch in range(EPOCH):
        avg_loss = 0
        cnt = 0
        for step, (images, labels) in enumerate(trainLoader):
            images = images.cuda()
            labels = labels.cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            _, outputs = model(images)
            loss = cost(outputs, labels)
            avg_loss += loss.data
            cnt += 1
            if step % 50 == 0:
                print("[%s E: %d] loss: %f, avg_loss: %f" % (modelname,epoch, loss.data, avg_loss / cnt))
            loss.backward()
            optimizer.step()
        scheduler.step(avg_loss)
        # save model every epoch
        torch.save(model.state_dict(), './SaveModelParams/' + str(modelname) + '.pkl-epoch' + str(epoch))

    # Save the Trained Model
    torch.save(model.state_dict(), './SaveModelParams/' + str(modelname) + '.pkl-final')
