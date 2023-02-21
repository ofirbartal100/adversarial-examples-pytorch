import torch.nn as nn
import torch
import torch.nn.functional as F
from adversarial_examples_pytorch.adv_gan.pix2pix_networks import NLayerDiscriminator

class Discriminator_MNIST(nn.Module):
	def __init__(self):
		super(Discriminator_MNIST, self).__init__()

		self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
		#self.in1 = nn.InstanceNorm2d(8)
		# "We do not use instanceNorm for the first C8 layer."

		self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
		self.in2 = nn.InstanceNorm2d(16)

		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		self.in3 = nn.InstanceNorm2d(32)

		self.fc = nn.Linear(3*3*32, 1)

	def forward(self, x):

		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
		x = F.leaky_relu(self.in2(self.conv2(x)), negative_slope=0.2)

		x = F.leaky_relu(self.in3(self.conv3(x)), negative_slope=0.2)

		x = x.view(x.size(0), -1)

		x = self.fc(x)

		return x

# class Discriminator_CIFAR10(Discriminator_MNIST):
# 	def __init__(self):
# 		super().__init__()
# 		self.conv1 = nn.Conv2d(3,8, kernel_size=4, stride=2, padding=1) # 3 channels
# 		self.fc = nn.Linear(4*4*32, 1)

class Discriminator_CIFAR10(NLayerDiscriminator):
	def __init__(self):
		super().__init__(3, 64, 2, nn.BatchNorm2d)


if __name__ == '__main__':

	from torchscope import scope
	# from tensorboardX import SummaryWriter
	from torch.autograd import Variable
	from torchvision import models

	# X = Variable(torch.rand(13, 1, 28, 28))

	model = Discriminator_CIFAR10()
	scope(model,input_size=(3,32,32))
	# model = Discriminator_MNIST()
	# model(X)

	# with SummaryWriter(log_dir="visualization/Discriminator_MNIST", comment='Discriminator_MNIST') as w:
		# w.add_graph(model, (X, ), verbose=True)

# CIFAR-10 Discriminator
# ------------------------------------------------------------------------------------------------------
#         Layer (type)               Output Shape          Params           FLOPs           Madds
# ======================================================================================================
#             Conv2d-1            [2, 64, 16, 16]           3,136         802,816       1,572,864
#          LeakyReLU-2            [2, 64, 16, 16]               0          16,384               0
#             Conv2d-3             [2, 128, 8, 8]         131,072       8,388,608      16,769,024
#        BatchNorm2d-4             [2, 128, 8, 8]             256          16,384          32,768
#          LeakyReLU-5             [2, 128, 8, 8]               0           8,192               0
#             Conv2d-6             [2, 256, 7, 7]         524,288      25,690,112      51,367,680
#        BatchNorm2d-7             [2, 256, 7, 7]             512          25,088          50,176
#          LeakyReLU-8             [2, 256, 7, 7]               0          12,544               0
#             Conv2d-9               [2, 1, 6, 6]           4,097         147,492         294,912
# ======================================================================================================
# Total params: 663,361
# Trainable params: 663,361
# Non-trainable params: 0
# Total FLOPs: 35,107,620
# Total Madds: 70,087,424
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.36
# Params size (MB): 0.63
# Estimated Total Size (MB): 1.00
# FLOPs size (GB): 0.04
# Madds size (GB): 0.07
# ----------------------------------------------------------------