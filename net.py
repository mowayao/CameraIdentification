#coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torchvision.models import resnet101, resnet50, resnet34, resnet18, resnet152, densenet161, densenet201, densenet121, densenet169

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1)).cuda()
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask).cuda()
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p



        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class ResNet(nn.Module):
    def __init__(self, dep, num_classes):
        super(ResNet, self).__init__()
        expansion = 1
        if dep == 18:
            net = resnet18(pretrained=True)
        elif dep == 34:
            net = resnet34(pretrained=True)
        elif dep == 50:
            net = resnet50(pretrained=True)
            expansion = 4
        elif dep == 101:
            net = resnet101(pretrained=True)
            expansion = 4
        else:
            net = resnet152(pretrained=True)
            expansion = 4
        self.features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.fc_input = expansion * 512
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        _, _, h, w = x.size()
        x = F.avg_pool2d(x, kernel_size=(h, w))
        x = x.view(-1, self.fc_input)
        x = self.fc(x)
        return x


class IncepResnet(nn.Module):
    def __init__(self, num_classes):
        super(IncepResnet, self).__init__()
        net = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        self.features = nn.Sequential(
            net.conv2d_1a,
            net.conv2d_2a,
            net.conv2d_2b,
            net.maxpool_3a,
            net.conv2d_3b,
            net.conv2d_4a,
            net.maxpool_5a,
            net.mixed_5b,
            net.repeat,
            net.mixed_6a,
            net.repeat_1,
            net.mixed_7a,
            net.repeat_2,
            net.block8,
            net.conv2d_7b
        )
        self.avg_pool = net.avgpool_1a
        self.fc = nn.Sequential(
            nn.Linear(1536, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 1536)
        x = self.fc(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, ver, num_classes, p=0):
        super(DenseNet, self).__init__()
        if ver == 161:
            net = densenet161(pretrained=True)
            layers = [nn.Linear(2208, 512), nn.Linear(512, num_classes)]
            if p > 0:
                layers.insert(1, nn.Dropout(p))
            self.fc = nn.Sequential(*layers)
        elif ver == 121:
            net = densenet121(pretrained=True)
            layers = [nn.Linear(1024, 512), nn.Linear(512, num_classes)]
            if p > 0:
                layers.insert(1, nn.Dropout(p))
            self.fc = nn.Sequential(*layers)
        elif ver == 169:
            net = densenet161(pretrained=True)
            layers = [nn.Linear(1664, 512), nn.Linear(512, num_classes)]
            if p > 0:
                layers.insert(1, nn.Dropout(p))
            self.fc = nn.Sequential(*layers)
        elif ver == 201:
            net = densenet201(pretrained=True)
            #layers = [nn.Linear(1920, num_classes)]
            layers = [
                nn.Linear(1920, 512),
                nn.ReLU(True),
                #nn.Dropout(p),
                nn.Linear(512, num_classes)]
            self.fc = nn.Sequential(*layers)
        self.features = net.features

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = F.relu(x, inplace=True)
        _, _, h, w = x.size()
        x = F.avg_pool2d(x, kernel_size=(h, w))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    net = IncepResnet(10)
    import numpy as np
    from torch.autograd import Variable

    a = torch.FloatTensor(np.ones((1, 3, 299, 299)))

    a = Variable(a).cuda()

    net.cuda()

    print net(a).size()
