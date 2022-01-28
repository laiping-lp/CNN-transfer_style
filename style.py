from __future__ import print_function
from mimetypes import init
from unicodedata import name
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
from PIL import Image
import matplotlib.pyplot as plt
 
import torchvision.transforms as transforms
import torchvision.models as models
 
import copy
import glob
import os

# 来判断是否有可用的GPU，代表torch.tensor分配的设备cuda(gpu)/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# 输出图像的所需尺寸
# 如果没有GPU，请使用小尺寸
imsize = 512 if torch.cuda.is_available() else 128                     
#一般输入深度网络的特征图长宽是相等的，就不能采取等比例缩放的方式了，需要同时指定长宽
loader = transforms.Compose([
    transforms.Resize(imsize),                                    # 缩放导入的图像
    transforms.ToTensor()])                                # 将其转换为torch tensor

def image_loader(image_name):
# Image.open()函数只是保持了图像被读取的状态，但是图像的真实数据并未被读取，
# 因此如果对需要操作图像每个元素，如输出某个像素的RGB值等，需要执行对象的load()方法读取数据。
    image = Image.open(image_name).convert('RGB')
    #unsqueeze是增加一个维度
    image = loader(image).unsqueeze(0)                # 需要伪造的批次尺寸以适合网络的输入尺寸
    return image.to(device, torch.float)

unloader = transforms.ToPILImage() 

def imshow(tensor, title=None):
    image = tensor.cpu().clone()                       # 我们克隆张量不对其进行更改
    #squeeze是减少一个维度
    image = image.squeeze(0)                           # 删除假批次尺寸
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)          

class ContentLoss(nn.Module):
 
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
 
        # 我们将目标内容与所使用的树“分离”
        # 动态计算梯度：这是一个规定值，
        # 不是变量。 否则，准则的前进方法
        # 将引发错误。
        self.target = target.detach()
 
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
 
 
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=特征图数量
    # (c,d)=dimensions of a f. map (N=c*d)
 
    features = input.view(a * b, c * d)  # 将FXML调整为\ hat FXML
 
    G = torch.matmul(features, features.t())
 
    # 我们将gram矩阵的值“规范化”
    # 除以每个要素图中的元素数量。
    return G.div(a * b * c * d)
 
 
class StyleLoss(nn.Module):
 
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
 
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# 创建一个模块来标准化输入图像，以便我们可以轻松地将其放入
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # 查看均值和标准差以使其为[C x 1 x 1]，以便它们可以
        # 直接使用形状为[B x C x H x W]的图像张量。
        # B是批量大小。 C是通道数。 H是高度，W是宽度。
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
 
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
 
    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# 所需的深度层以计算样式/内容损失：
content_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
 
    # 标准化模块
    normalization = Normalization(normalization_mean, normalization_std).to(device)
 
    # 只是为了获得对内容/样式的可迭代访问或列表
    # losses
    content_losses = []
    style_losses = []
 
    # 假设cnn是nn.Sequential，那么我们创建一个新的nn.Sequential
    # 放入应该顺序激活的模块
    model = nn.Sequential(normalization)
 
    i = 0  # 每当转换时就增加
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):              #如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # 旧版本与我们在下面插入的ContentLoss和StyleLoss不能很好地配合使用。
            # 因此，我们在这里替换为不适当的。
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
 
        model.add_module(name, layer)
 
        if name in content_layers:
            # 增加内容损失:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
 
        if name in style_layers:
            # 增加样式损失:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
 
    # 现在我们在最后一次内容和样式丢失后修剪图层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
 
    model = model[:(i + 1)]
 
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # 此行显示输入是需要渐变的参数
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
 
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    # """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img)
    optimizer = get_input_optimizer(input_img)
 
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
 
        def closure():
            # 更正更新后的输入图像的值
            input_img.data.clamp_(0, 1)
 
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
 
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
 
            style_score *= style_weight
            content_score *= content_weight
 
            loss = style_score + content_score
            loss.backward()
 
            run[0] += 1
            if run[0] % 50 == 0:
                print("run [{0}/{1}]:".format(run[0],num_steps))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
 
            return style_score + content_score
 
        optimizer.step(closure)
 
    # 最后更正...
    input_img.data.clamp_(0, 1)
 
    return input_img

if __name__ == '__main__':
    #内容图像的路径
    file_content_path = r"images\content\\"
    #风格图像的路径
    file_style_path = r"images\style\\"
    #保存生成图像的路径
    file_output_path = r"images\output\\"
    #所有风格图像的名字
    image_style_name = os.listdir(file_style_path)
    # print(image_style_name)
    #所有内容图像的名字
    image_content_name = os.listdir(file_content_path)
    #读取每一张风格图像的路径
    images_style_path = glob.glob(os.path.join(file_style_path + '*.jpg'))
    #读取每一张内容图像的路径
    images_content_path = glob.glob(os.path.join(file_content_path + '*.jpg'))  # 所有图片路径
    #外层循环是循环每一张风格图像
    for style_name in image_style_name:
        style_img = image_loader(file_style_path+style_name)
        i = style_name.rfind('.')
        styleName = style_name[:i]
        print(styleName)
        #内层循环是循环每一张内容图像
        for content_name in image_content_name:
            j = content_name.rfind('.')
            contentName = content_name[:j]
            content_img = image_loader(file_content_path+content_name)
            print(styleName," ",contentName)
            #同一张画布同时显示风格图像和内容图像
            plt.figure()
            plt.subplot(1,2,1)
            imshow(style_img, title='Style Image')
            plt.subplot(1,2,2)
            imshow(content_img, title='Content Image')
            plt.show(block=False)
            plt.pause(2)
            plt.close()
            # break
            #VGG19网络
            cnn = models.vgg19(pretrained=True).features.to(device).eval()
            
            cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            
            cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
            input_img = content_img.clone()
            # 如果要使用白噪声，请取消注释以下行：
            # input_img = torch.randn(content_img.data.size(), device=device)
            #在同一张画布显示风格图像，内容图像，生成图像
            plt.figure()
            plt.subplot(1,3,1)
            imshow(input_img, title='Input Image') 

            output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,content_img, style_img, input_img)
            # 将原始输入图像添加到图中：

            plt.subplot(1,3,2)
            imshow(style_img, title='Style Image')
            plt.subplot(1,3,3)
            imshow(output, title='Output Image')
            #保存生成图像至相应路径
            plt.savefig(file_output_path+styleName+'_{}.jpg'.format(contentName))
            plt.show(block=False)
            plt.pause(2)
            plt.close() 

            # sphinx_gallery_thumbnail_number = 4
            plt.ioff()
            plt.show()
            break
        break