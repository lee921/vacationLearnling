import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import LoadData
import VGG16_model as VGG16

BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 50
N_CLASSES = 10

# load train datasets and test datasets
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪224×224
    transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
    transforms.ToTensor(),  # 将给定图像转为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化处理
                         std=[0.229, 0.224, 0.225]),
])

datasets_path = 'E:\PythonProject\datasets\CIFAR-10\cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = LoadData.load_CIFAR10(datasets_path)
trainData = dsets.ImageFolder('../data/imagenet/train', transform)
testData = dsets.ImageFolder('../data/imagenet/test', transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

# load vgg-16 model
vgg16 = VGG16.VGG16(n_classes=N_CLASSES)
vgg16.cuda()

# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# Train the model
for epoch in range(EPOCH):
    avg_loss = 0
    cnt = 0
    for images, labels in trainLoader:
        images = images.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, outputs = vgg16(images)
        loss = cost(outputs, labels)
        avg_loss += loss.data
        cnt += 1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
        loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)

# Test the model
vgg16.eval()
correct = 0
total = 0

for images, labels in testLoader:
    images = images.cuda()
    _, outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print(predicted, labels, correct, total)
    print("avg acc: %f" % (100 * correct / total))

# Save the Trained Model
torch.save(vgg16.state_dict(), 'VGG16.pkl')
