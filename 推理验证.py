# 黑字白底的图片  图片需要反转   mnist数据集是黑底白字
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from LeNet_mnist_New import ConvNet as net


def load_model(model_path):
    # 加载训练好的模型
    model = net()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = ImageOps.invert(img)  # 反转图像颜色
    img_transformed = transform(img)
    return img_transformed


def predict(model, img_transformed):
    # 推理
    img_transformed = img_transformed.view(1, 1, 28, 28)  # 调整形状以匹配模型输入
    with torch.no_grad():
        output = model(img_transformed)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()


# 主程序
model = load_model('Lenet_model_TrainAcc97.0_TestAcc97.4.pth')
img_transformed = preprocess_image('手写数字_3.jpg')

# 绘制预处理后的图像
plt.imshow(img_transformed.squeeze(), cmap='gray')  # 使用squeeze去除批次维度以便显示
plt.title('Preprocessed Image')
plt.show()

predicted_digit = predict(model, img_transformed)
print(f'预测的数字: {predicted_digit}')
