from mymodel import vgg16 as VGG16
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from mymodel import resnet18 as ResNet

N_CLASSES = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# modelname in {vgg16,resnet18}
modelname = 'resnet18'
# inSize vgg16:32, resnet18:32
inSize = 32

testSetPath = 'E:\PythonProject\datasets\CIFAR-10'
saveParamsPath = './SaveModelParams/' + modelname + '.pkl'

# load data
transform = transforms.Compose([
    transforms.RandomResizedCrop(inSize),  # 随机裁剪224×224
    transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
    transforms.ToTensor(),  # 将给定图像转为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化处理
                         std=[0.229, 0.224, 0.225]),
])
testData = dsets.CIFAR10(root=testSetPath, train=False, download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=10, shuffle=False)


def getModel(modelname='vgg16'):
    model = None
    if modelname == 'vgg16':
        # load vgg-16 model
        model = VGG16.VGG16(inSize=inSize, n_classes=N_CLASSES)
    if modelname == 'resnet18':
        model = ResNet.ResNet(inSize=inSize,num_classes=N_CLASSES)

    model.cuda()
    model.eval()
    model.load_state_dict(torch.load(saveParamsPath))
    return model


model = getModel(modelname)
correct = 0
total = 0

for images, labels in testLoader:
    images = images.cuda()
    _, outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print('predict:', predicted.cpu().numpy(), 'labels:', labels.cpu().numpy(), 'correct:', correct.cpu().numpy(), 'total:', total)
    print("avg acc: %f" % (100 * float(correct) / float(total)))
