import torch
import torch.nn as nn
from torch.autograd import Variable

def C_BN_ACT(c_in, c_out, activation, transpose=False, dropout=None, bn=True):
    layers = []
    if transpose:
        layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False))
    else:
        layers.append(         nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(activation)
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    '''
    Input: (batch_size, num_channels, H, W)
    Output: (batch_size, 512, H / 2**7, W / 2**7)
    '''
    def __init__(self,k_list):
        super(Encoder, self).__init__()
        activation = nn.LeakyReLU(0.2)
        layers = []
        for i in range(1, len(k_list)):
            c_in, c_out = k_list[i - 1], k_list[i]
            bn = False if i == len(k_list) - 1 else True
            layers.append(C_BN_ACT(c_in, c_out, activation, bn=bn))
        self.convs = nn.Sequential(*layers)
    
    def forward(self, x):
        Ex = self.convs(x)
        return Ex

class Decoder(nn.Module):
    '''
    Input: (batch_size, 512, H, W), (batch_size, attr_dim)
    Output: (batch_size, 3, H * 2**7, W * 2**7)
    '''
    def __init__(self, k_list, attr_dim, image_size=256, num_channels=3):
        super(Decoder, self).__init__()
        activation = nn.ReLU()
        
        self.image_size = image_size
        if self.image_size == 256:
            self.deconv1 = C_BN_ACT(k_list[7] + attr_dim, k_list[6], activation, transpose=True)
        self.deconv2 = C_BN_ACT(k_list[6] + attr_dim, k_list[5], activation, transpose=True)
        self.deconv3 = C_BN_ACT(k_list[5] + attr_dim, k_list[4], activation, transpose=True)
        self.deconv4 = C_BN_ACT(k_list[4] + attr_dim, k_list[3], activation, transpose=True)
        self.deconv5 = C_BN_ACT(k_list[3] + attr_dim, k_list[2], activation, transpose=True)
        self.deconv6 = C_BN_ACT(k_list[2] + attr_dim, k_list[1], activation, transpose=True)
        self.deconv7 = C_BN_ACT(k_list[1] + attr_dim, k_list[0], nn.Tanh(), transpose=True, bn=False)
        
    def repeat_concat(self, Ex, attrs):
        H, W = Ex.size()[2], Ex.size()[3]
        attrs_ = attrs.repeat(H, W, 1, 1).permute(2, 3, 0, 1)
        Ex_ = torch.cat([Ex, attrs_], dim=1)
        return Ex_
        
    def forward(self, Ex, attrs):
        if self.image_size == 256:
            Ex = self.deconv1(self.repeat_concat(Ex, attrs))
        Ex = self.deconv2(self.repeat_concat(Ex, attrs))
        Ex = self.deconv3(self.repeat_concat(Ex, attrs))
        Ex = self.deconv4(self.repeat_concat(Ex, attrs))
        Ex = self.deconv5(self.repeat_concat(Ex, attrs))
        Ex = self.deconv6(self.repeat_concat(Ex, attrs))
        Ex = self.deconv7(self.repeat_concat(Ex, attrs))
        return Ex


class Discriminator(nn.Module):
    '''
    Input: (batch_size, 512, H / 2**7, W / 2**7)
    Output: (batch_size, num_attrs)
    '''
    def __init__(self, num_attrs, image_size=256):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        if image_size == 256:
            self.conv = C_BN_ACT(512, 512, nn.LeakyReLU(0.2)) # ReLU? Dropout?
        self.fc1 = nn.Linear(512, 512)
        self.dp1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_attrs)
        self.dp2 = nn.Dropout(0.3)
    
    def forward(self, Ex):
        if self.image_size == 256:
            Ex = self.conv(Ex)
        p = Ex.view(Ex.size()[0], Ex.size()[1])
        p = self.dp1(self.fc1(p))
        p = self.dp2(self.fc2(p))
        return p