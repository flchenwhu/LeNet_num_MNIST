import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary  #输出网络模型结构的相关包
from torchvision import datasets, transforms
from matplotlib import pyplot as plt # 导入matplotlib库，用于绘图
from utils import plot_image,plot_curve,one_hot # 导入自定义的一些工具函数，如plot_image, plot_curve, one_hot

from torchmetrics import Accuracy  ##输出训练Acc，方案二需要的包
# import os  # 导入os库，用于操作系统相关的功能
# os.environ['KMP_DUPLICATE_LIB_OK']='True'   # 设置一个环境变量，防止出现重复的库的错误

# torch.__version__
# print(torch.__version__)

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

# 定义卷积神经网络
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




# 训练
def train(model, device, train_loader, optimizer, epoch):
    train_loss=[] # 初始化一个空列表，用来存储每个批次的训练损失值
    model.train() # 将模型设置为训练模式，启用梯度计算和dropout等
    for batch_idx, (data, target) in enumerate(train_loader): # 遍历训练数据集的每个批次，batch_idx是批次的索引，data是输入数据，target是标签
        data, target = data.to(device), target.to(device) # 将数据和标签移动到指定的设备上，比如GPU或CPU
        optimizer.zero_grad() # 将优化器中的梯度清零，避免累积
        output = model(data) # 将输入数据传入模型，得到输出结果
        loss = F.nll_loss(output, target) # 计算输出结果和标签之间的负对数似然损失函数
        loss.backward() # 反向传播，计算损失函数对模型参数的梯度
        optimizer.step() # 更新模型参数，根据梯度和学习率等
        train_loss.append(loss.item()) # 将当前批次的损失值转换为Python数值并整体添加到train_loss列表末尾
        if(batch_idx+1)%30 == 0: # 每隔30个批次打印一次训练信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( # 打印当前周期数，已处理的数据量，总数据量，进度百分比，当前批次的损失值
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    ###############方案一：直接计算  输出训练准确度
    # correct = 0
    # for data, target in train_loader:
    #     data, target = data.to(device), target.to(device)
    #     output = model(data)
    #     pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
    #     correct += pred.eq(target.view_as(pred)).sum().item()
    #     acc =100. * correct / len(train_loader.dataset)
    #     # print('epoch:',epoch,'迭代次数:',batch_idx,'loss:',loss.item(),'Train accuracy:', acc)
    # print('epoch: {} 迭代次数: {} loss: {:.3f} Train accuracy: {}/{} ({:.0f}%)\n'.format(epoch, batch_idx, loss.item(), correct, len(train_loader.dataset), acc))
    # return acc, train_loss

    ###############方案二：指标计算函数Accuracy 来输出训练准确度    
    accuracy = Accuracy(task='multiclass', num_classes=10) # 创建一个Accuracy对象，用来计算多分类任务的准确度，指定类别数为10
    for data, target in train_loader: # 遍历训练数据集的每个批次，data是输入数据，target是标签
        data, target = data.to(device), target.to(device) # 将数据和标签移动到指定的设备上，比如GPU或CPU
        output = model(data) # 将输入数据传入模型，得到输出结果
        # 更新指标
        accuracy(output, target) # 用输出结果和标签更新Accuracy对象的内部状态
    # 获取准确度
    Acc = accuracy.compute() # 调用Accuracy对象的compute方法，计算并返回整个训练集的准确度
    print('epoch: {} batch: {} loss: {:.3f} Train accuracy: {:.0f}%\n'.format(epoch, batch_idx, loss.item(), Acc * 100)) # 打印当前周期数，最后一个批次的损失值，训练集的准确度
    return Acc, train_loss # 返回训练集的准确度和每个批次的损失值列表


# 测试
def test(model, device, test_loader): # 定义一个测试函数，接受模型，设备和测试数据集作为参数
    model.eval() # 将模型设置为评估模式，关闭梯度计算和dropout等
    test_loss = 0 # 初始化一个变量，用来累计测试集的损失值
    correct = 0 # 初始化一个变量，用来累计测试集的正确预测数
    with torch.no_grad(): # 关闭梯度计算，节省内存和时间
        for data, target in test_loader: # 遍历测试数据集的每个批次，data是输入数据，target是标签
            data, target = data.to(device), target.to(device) # 将数据和标签移动到指定的设备上，比如GPU或CPU
            output = model(data) # 将输入数据传入模型，得到输出结果
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加到test_loss变量上，使用负对数似然损失函数，并将reduction参数设置为'sum'，表示对一批的损失求和，而不是求平均
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标作为预测结果，output.max(1, keepdim=True)返回一个元组，包含每行的最大值和对应的下标，取[1]表示只取下标
            correct += pred.eq(target.view_as(pred)).sum().item() # 将预测结果和标签进行比较，如果相等则返回True，否则返回False，然后对比较结果求和，得到一批的正确预测数，加到correct变量上

    test_loss /= len(test_loader.dataset) # 将累计的测试集损失除以测试集的总样本数，得到平均损失
    Acc =100. * correct / len(test_loader.dataset) # 将累计的正确预测数除以测试集的总样本数，得到准确率，并乘以100转换为百分比
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(test_loader.dataset),Acc))  # 打印测试集的平均损失，正确预测数，总样本数和准确率
    return Acc # 返回测试集的准确率



if __name__ == '__main__': # 这是一个常用的Python语句，用于判断当前模块是否是主程序，如果是，则执行以下代码
    Train_Loss = [] # 定义一个空列表，用于存储训练过程中的损失(loss)值
    model = ConvNet().to(DEVICE) # 创建一个ConvNet类的实例，并将其转移到指定的设备(DEVICE)上，可以是CPU或GPU
    summary(model, input_size=(1, 28, 28)) # 调用summary函数，输入模型和输入大小，输出网络结构和参数数量
    optimizer = optim.Adam(model.parameters()) # 创建一个优化器(optimizer)，使用Adam算法，输入模型的参数
    for epoch in range(1, EPOCHS + 1): # 开始一个循环，从1到EPOCHS(预定义的训练轮数)，每次循环代表一轮训练
        train_acc, Trainloss =train(model, DEVICE, train_loader, optimizer, epoch) # 调用train函数，输入模型，设备，训练数据加载器(train_loader)，优化器和当前轮数，输出训练精度(train_acc)和训练损失(Trainloss)
        Train_Loss.extend(Trainloss) # 将Trainloss添加到Train_Loss列表中
    Test_Acc =test(model, DEVICE, test_loader) # 调用test函数，输入模型，设备和测试数据加载器(test_loader)，输出测试精度(Test_Acc)
    plot_curve(Train_Loss) # 调用plot_curve函数，输入Train_Loss列表，输出损失变化曲线图


####################模型保存 保存整个网络
torch.save({
            'epoch': EPOCHS,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'Testacc': Test_Acc,
            'TrainAcc': train_acc,
            'Batchsize': BATCH_SIZE,
            }, f"Lenet数字识别_Batchsize{BATCH_SIZE}_epoch{EPOCHS}_TrainAcc{train_acc:.4f}_TestAcc{Test_Acc:.4f}.pth")
# 文件名是一个格式化字符串(f-string)，包含模型名称(Lenet数字识别)，批量大小(Batchsize)，训练轮数(epoch)，训练精度(TrainAcc)和测试精度(TestAcc)，以及文件扩展名(.pth)
# 读取 use_model = torch.load('./Train_model.pt')
#     torch.save(model, './Lent_train_model.pth')
