import  torch
from    matplotlib import pyplot as plt


def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['loss'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()
    # plt.draw()  # 自己修改



def plot_image(img, label, name):

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    # plt.draw()  # 自己修改

def one_hot(label, depth=10):
    #out = torch.zeros(label.size(0), depth)
    out=torch.zeros(label.size(0),depth,device=torch.device('cuda:0'))  # 自己修改 加了cuda()
    # idx = torch.LongTensor(label).view(-1, 1)
    idx=torch.as_tensor(label, device=torch.device('cuda:0')).view(-1,1)  # 自己修改 加了cuda()
    # print(idx.is_cuda, out.is_cuda)  # 自己修改 验证变量是否加载到cuda
    out.scatter_(dim=1, index=idx, value=1)
    return out