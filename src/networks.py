import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import torch.nn.functional as F



class G_Net(nn.Module):
    def __init__(self, input_channels, residual_blocks, threshold):
        super(G_Net, self).__init__()

        # Encoder
        self.encoder_1 = nn.Sequential(nn.Conv2d(input_channels + 1, 64, 7, 1, 3), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
        self.encoder_2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.encoder_3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.encoder_4 = nn.Sequential(nn.Conv2d(256, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True))


        # Middle
        bolock_1 = GDB(256)
        bolock_2 = GDB(256)
        bolock_3 = GDB(256)
        bolock_4 = GDB(256)
        blocks = [bolock_1, bolock_2, bolock_3, bolock_4]

        self.middle = nn.Sequential(*blocks)


        # Decoder
        self.decoder_1 = nn.Sequential(nn.Conv2d(256, 256*4, 3, 1, 1), nn.PixelShuffle(2),nn.InstanceNorm2d(256),nn.LeakyReLU(0.2, inplace=True))
        self.decoder_2 = nn.Sequential(nn.Conv2d(256, 128*4, 3, 1, 1), nn.PixelShuffle(2),nn.InstanceNorm2d(128),nn.LeakyReLU(0.2, inplace=True))
        self.decoder_3 = nn.Sequential(nn.Conv2d(128, 64*4, 3, 1, 1), nn.PixelShuffle(2),nn.InstanceNorm2d(64),nn.LeakyReLU(0.2, inplace=True))
        self.decoder_4 = nn.Conv2d(64, 3, 7, 1, 3)
        self.attention = Comp_Attn(256)

    def encoder(self, x, mask):
        x = torch.cat([x, mask], 1)
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        return x_4
    
    def decoder(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        return x
        

    def forward(self, x, mask):
        gt = x
        x = (x * (1 - mask).float()) + mask
        # input mask: 1 for hole, 0 for valid
        x = self.encoder(x, mask)
        x = self.middle(x)
        x, attention = self.attention(x)
        x = self.decoder(x)

        x = (torch.tanh(x) + 1) / 2
        return x


# original D
class D_Net(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(D_Net, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

    

class Comp_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Comp_Attn,self).__init__()
        
        self.q_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.k_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.v_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        
        self.com_conv_1 = nn.Conv2d(in_channels = in_dim//8 , out_channels = in_dim//8 , kernel_size= 1)
        self.com_conv_2 = nn.Conv2d(in_channels = in_dim//8 , out_channels = in_dim//8 , kernel_size= 1)
        self.com_conv_3 = nn.Conv2d(in_channels = in_dim//8 , out_channels = in_dim//8 , kernel_size= 1)
        self.com_conv_4 = nn.Conv2d(in_channels = in_dim//8 , out_channels = in_dim//8 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        
        b, c, w, h = x.size()
        proj_q = self.q_conv(x)
        proj_k = self.k_conv(x)
        
        re_q = self.com_conv_1(proj_q).view(b,-1,w*h).permute(0,2,1) # B*N*C
        co_q = self.com_conv_2(proj_q).view(b,-1,w*h).permute(0,2,1) # B*N*C
        re_k = self.com_conv_3(proj_k).view(b,-1,w*h)
        co_k = self.com_conv_4(proj_k).view(b,-1,w*h)
        energy = torch.bmm(re_q,re_k) + torch.bmm(co_q,co_k)
        attention = self.softmax(energy) # B*N*N
        proj_v = self.v_conv(x).view(b,-1,w*h) # B*C*N

        out = torch.bmm(proj_v,attention.permute(0,2,1) )
        out = out.view(b,c,w,h)
        
        out = self.gamma*out + x
        
#         y = self.avg_pool(x).view(b, c, 1, 1)
#         y = self.fc(y)
#         y = y.expand_as(x)
        
#         out = out + x*y
        return out,attention


class MSFB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MSFB, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_5 = nn.Sequential(nn.Conv2d(channel, channel, 5, 1, 2), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_7 = nn.Sequential(nn.Conv2d(channel, channel, 7, 1, 3), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_1_1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_1_2 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_mid = nn.Sequential(nn.Conv2d(2*channel, channel, 3, 1, 1), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_mid_1 = nn.Sequential(nn.Conv2d(2*channel, channel, 3, 1, 1), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_mid_2 = nn.Sequential(nn.Conv2d(2*channel, channel, 3, 1, 1), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.conv_last = nn.Sequential(nn.Conv2d(4*channel, channel, 1, 1, 0), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        self.activation = nn.LeakyReLU(0.2, True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_3 = self.conv_3(x)
        x_5 = self.conv_5(x)
        x_7 = self.conv_7(x)
        
        x_13 = self.conv_mid(torch.cat([x_1, x_3], 1))
        x_5 = self.conv_1_1(x_5)
        
        x_135 = self.conv_mid_1(torch.cat([x_13, x_5], 1))
        x_7 = self.conv_1_2(x_7)
        
        x_1357 = self.conv_mid_2(torch.cat([x_135, x_7], 1))
        
        y_1 = x_1
        y_2 = x_13
        y_3 = x_135
        y_4 = x_1357
        y_mid = torch.cat([y_1, y_2, y_3, y_4], 1)
        y_mid = self.conv_last(y_mid)
        
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x + y_mid * y.expand_as(y_mid)

class GDB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GDB, self).__init__()
        self.base_block_1 = MSFB(channel)
        self.base_block_2 = MSFB(channel)
        self.base_block_3 = MSFB(channel)
        self.base_block_4 = MSFB(channel)
        self.fuse = nn.Sequential(nn.Conv2d(channel*4, channel, 1, 1, 0), nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2, True))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_1 = self.base_block_1(x)
        x_2 = self.base_block_2(x + x_1)
        x_3 = self.base_block_3(x + x_1 + x_2)
        x_4 = self.base_block_4(x + x_1 + x_2 + x_3)
        y_mid = torch.cat([x_1, x_2, x_3, x_4], 1)
        y_mid = self.fuse(y_mid)
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        
        return x + y_mid * y.expand_as(y_mid)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


if __name__ == '__main__':
    print("No Abnormal!")
