import torch
import torch.nn as nn
import torch.nn.functional as F
from adversarial_examples_pytorch.adv_gan.pix2pix_networks import UnetSkipConnectionBlock

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))

        out = out + residual

        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)

        padding = kernel_size // 2

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):

        if self.upsample:
            x = self.upsample_layer(x)

        x = self.conv2d(x)

        return x



class Generator_MNIST(nn.Module):
    def __init__(self):
        super(Generator_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(32)

        self.resblock1 = ResidualBlock(32)
        self.resblock2 = ResidualBlock(32)
        self.resblock3 = ResidualBlock(32)
        self.resblock4 = ResidualBlock(32)


        self.up1 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(16)
        self.up2 = UpsampleConvLayer(16, 8, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(8)


        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.in6 = nn.InstanceNorm2d(8)


    def forward(self, x):

        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = F.relu(self.in4(self.up1(x)))
        x = F.relu(self.in5(self.up2(x)))

        x = self.in6(self.conv4(x)) # remove relu for better performance and when input is [-1 1]

        return x

# class Generator_CIFAR10(Generator_MNIST):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1) # 3 channels
#         self.conv4 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1) # 3 channels

class UnetGeneratorCIFAR(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self,x):
        return self.model(x)

class Generator_CIFAR10(UnetGeneratorCIFAR):
    def __init__(self):
        super().__init__(3, 3,  16, nn.BatchNorm2d)



if __name__ == '__main__':

    # from tensorboardX import SummaryWriter
    from torch.autograd import Variable
    from torchvision import models
    from torchscope import scope

    # X = Variable(torch.rand(13, 3, 32, 32))

    model = Generator_CIFAR10()
    scope(model,input_size=(3,32,32))

    # with SummaryWriter(log_dir="tmp/Generator_MNIST", comment='Generator_MNIST') as w:
    #     w.add_graph(model, (X, ))
# CIFAR-10 Generator
# ------------------------------------------------------------------------------------------------------
#         Layer (type)               Output Shape          Params           FLOPs           Madds
# ======================================================================================================
#             Conv2d-1            [2, 16, 16, 16]             768         196,608         389,120
#          LeakyReLU-2            [2, 16, 16, 16]               0           4,096               0
#             Conv2d-3              [2, 64, 8, 8]          16,384       1,048,576       2,093,056
#        BatchNorm2d-4              [2, 64, 8, 8]             128           8,192          16,384
#          LeakyReLU-5              [2, 64, 8, 8]               0           4,096               0
#             Conv2d-6             [2, 128, 4, 4]         131,072       2,097,152       4,192,256
#        BatchNorm2d-7             [2, 128, 4, 4]             256           4,096           8,192
#          LeakyReLU-8             [2, 128, 4, 4]               0           2,048               0
#             Conv2d-9             [2, 128, 2, 2]         262,144       1,048,576       2,096,640
#              ReLU-10             [2, 128, 2, 2]               0             512             512
#   ConvTranspose2d-11             [2, 128, 4, 4]         262,144               0       2,096,640
#       BatchNorm2d-12             [2, 128, 4, 4]             256           4,096           8,192
#              ReLU-13             [2, 256, 4, 4]               0           4,096           4,096
#   ConvTranspose2d-14              [2, 64, 8, 8]         262,144               0       8,387,584
#       BatchNorm2d-15              [2, 64, 8, 8]             128           8,192          16,384
#              ReLU-16             [2, 128, 8, 8]               0           8,192           8,192
#   ConvTranspose2d-17            [2, 16, 16, 16]          32,768               0       4,193,280
#       BatchNorm2d-18            [2, 16, 16, 16]              32           8,192          16,384
#              ReLU-19            [2, 32, 16, 16]               0           8,192           8,192
#   ConvTranspose2d-20             [2, 3, 32, 32]           1,539               0         786,432
#              Tanh-21             [2, 3, 32, 32]               0               0               0
# ======================================================================================================
# Total params: 969,763
# Trainable params: 969,763
# Non-trainable params: 0
# Total FLOPs: 4,454,912
# Total Madds: 24,321,536
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.29
# Params size (MB): 0.92
# Estimated Total Size (MB): 1.21
# FLOPs size (GB): 0.00
# Madds size (GB): 0.02