import torch
import torch.nn as nn
import torch.nn.functional as F 
from config import *

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        import torch.nn.functional as F
        #nn.Conv2d(input channel, output channel, ...)
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size = 3,
                               stride=stride,
                               padding=1,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(planes) # Batchnorm은 사이의 가중치가 아니라 출력 층만 노말라이징
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size = 3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size = 1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes))
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()
        
        self.in_planes = 16
        # RGB여서 3, in_planes는 내맘대로 16
        self.conv1 = nn.Conv2d(1,16,
                               kernel_size = 3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64,num_classes)
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] *(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes,planes,stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = F.avg_pool2d(out4, 7)
        out6 = out5.view(out5.size(0), -1)
        out7 = self.linear(out6)
        
        return out7


"""Discriminator
    Description: CNN에 FC layer 붙여서 prediction을 냄
    Input: gray image (real + fake)
    Returns: 예측에 대한 Logit값 하나 반환 
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
                                   nn.BatchNorm2d(32), nn.LeakyReLU(0.2,inplace=True), )
        self.conv1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(64),nn.LeakyReLU(0.2,inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(128), nn.LeakyReLU(0.2,inplace=True), )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
                                    nn.BatchNorm2d(256), nn.LeakyReLU(0.2,inplace=True), )
        self.fc = nn.Linear(256*7*7, 1)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 256*7*7)
        x = self.fc(x)
        return x
    
"""Generator
    Description: Transposed CNN을 써서 noise를 이미지로 변환
    Input: Z_DIM 크기를 갖는 noise vector
    Returns: gray image 생성 
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = Z_DIM
        self.important_attr_dim = CLASSIFIER_ATTR_DIM
        self.fc1 = nn.Linear(64*7*7,CLASSIFIER_ATTR_DIM)
        self.fc2 = nn.Linear(CLASSIFIER_ATTR_DIM,CLASSIFIER_ATTR_DIM)
        self.fc3 = nn.Linear(CLASSIFIER_ATTR_DIM,CLASSIFIER_ATTR_DIM)
        self.fc4 = nn.Linear(CLASSIFIER_ATTR_DIM+Z_DIM, 256*7*7)
        self.up_conv0 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 1, padding = 1),
                                        nn.ReLU(), nn.BatchNorm2d(128))
        self.up_conv1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                                        nn.ReLU(), nn.BatchNorm2d(64))
        self.up_conv2 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1),
                                        nn.ReLU(), nn.BatchNorm2d(32))
        self.up_conv3 = nn.Sequential(nn.ConvTranspose2d(32, 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                                        )
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
    
    def forward(self, conv_block,latent):
        flatten = conv_block.view(-1,64*7*7)
        x = self.fc1(flatten)
        x = self.fc2(x)
        x = self.fc3(x)
        if x.shape[0] != latent.shape[0]: print(x.shape, latent.shape)
        concat = torch.cat([x,latent], dim=1)
        x = self.fc4(concat)
        x = x.view(-1, 256, 7, 7)
        x = self.up_conv0(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_conv3(x)
        x = torch.tanh(x)
        return x  

if __name__ == '__main__':
    print('Done!')
    print(Z_DIM)