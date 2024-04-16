# 命令行中运行tensorboard --logdir=runResult 来启动TensorBoard;  当前目录下  会有一个文件夹 runResult
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard的SummaryWriter

from matplotlib import pyplot as plt # 导入matplotlib库，用于绘图
from utils import plot_image,plot_curve,one_hot # 导入自定义的一些工具函数，如plot_image, plot_curve, one_hot

# 初始化TensorBoard的SummaryWriter
writer = SummaryWriter()

BATCH_SIZE=512 # 大概需要2G的显存
EPOCHS=2 # 总共训练批次(所有数据丢进去训练，为了实验方便，设置为2次)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

# 下载训练集 shujuji
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

# 下载测试集
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

# 定义卷积神经网络结构
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，10个filters 卷积核，核的大小5 filter
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # 全连接层 batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) #全连接层 batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总预测的数量
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        # 获取最大概率的预测结果
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    train_accuracy = 100. * correct / total
    # 每个批次结束后，记录损失和准确率
    # writer.add_scalar('Accuracy/train', 100. * correct / total, epoch * len(train_loader) + batch_idx)
    return train_accuracy, train_loss
# 测试函数
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将数据和标签移动到指定的设备上，比如GPU或CPU
            output = model(data)  # 将输入数据传入模型，得到输出结果
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # 将一批的损失相加到test_loss变量上，使用负对数似然损失函数，并将reduction参数设置为'sum'，表示对一批的损失求和，而不是求平均
            pred = output.max(1, keepdim=True)[
                1]  # 找到概率最大的下标作为预测结果，output.max(1, keepdim=True)返回一个元组，包含每行的最大值和对应的下标，取[1]表示只取下标
            correct += pred.eq(target.view_as(
                pred)).sum().item()  # 将预测结果和标签进行比较，如果相等则返回True，否则返回False，然后对比较结果求和，得到一批的正确预测数，加到correct变量上
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    # 记录测试损失和准确率
    # writer.add_scalar('Loss/test', test_loss, epoch)
    # writer.add_scalar('Accuracy/test', test_accuracy, epoch)
    return test_accuracy, test_loss

# 实例化网络、优化器和损失函数
model = ConvNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练和测试循环
for epoch in range(1, EPOCHS + 1):
    train_acc, _ = train(model, DEVICE, train_loader, optimizer, epoch)
    test_acc, _ = test(model, DEVICE, test_loader, epoch)
    # 将训练和测试准确率记录在同一幅图中
    writer.add_scalars('Accuracy', {'train': train_acc, 'test': test_acc}, epoch)

# 关闭SummaryWriter
writer.close()

####################模型保存 保存整个网络
torch.save({
            'epoch': EPOCHS,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'Testacc': test_acc,
            'TrainAcc': train_acc,
            'Batchsize': BATCH_SIZE,
            }, f"Lenet_model_TrainAcc{train_acc:.3f}_TestAcc{test_acc:.3f}.pth")
# 文件名是一个格式化字符串(f-string)，包含模型名称(Lenet数字识别)，批量大小(Batchsize)，训练轮数(epoch)，训练精度(TrainAcc)和测试精度(TestAcc)，以及文件扩展名(.pth)
