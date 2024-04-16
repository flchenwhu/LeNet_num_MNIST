import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 定义之前训练的网络结构
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

# 加载训练好的模型
model = ConvNet()
# 加载checkpoint
checkpoint = torch.load('Lenet_model_TrainAcc98.5_TestAcc98.5.pth', map_location=torch.device('cuda:0'))
# 检查'checkpoint'中是否有'model_state_dict'
if 'model_state_dict' in checkpoint:
    # 如果存在，就加载'model_state_dict'
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # 如果不存在，就直接加载'checkpoint'
    model.load_state_dict(checkpoint)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: 1.0 - x),  # 反转颜色  因为我们的图片是白底黑字
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载图像并进行预处理
img = Image.open('手写数字_推理2.jpg')

# 检查图像模式并转换
if img.mode == 'RGBA':
    img = img.convert('RGB')

img = ImageOps.invert(img)  # 反转图像颜色
img_transformed = transform(img)
# img_transformed = img_transformed.view(1, 28 * 28)  # 展平图像以匹配模型输入
img_transformed = img_transformed.unsqueeze(0)  # 添加一个批次维度，使其成为4维张量
# 绘制预处理后的图像
# plt.imshow(img_transformed.view(28, 28), cmap='gray')
plt.imshow(img_transformed.squeeze(), cmap='gray')  # 使用squeeze()移除单维度条目
plt.title('Preprocessed Image')
plt.show()

# 推理
with torch.no_grad():
    output = model(img_transformed)
    _, predicted = torch.max(output.data, 1)
    print(f'Predicted digit: {predicted.item()}')




