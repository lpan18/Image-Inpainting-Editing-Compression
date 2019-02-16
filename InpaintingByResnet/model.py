import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, inC = 4, outC = 3):
        super(ResNet, self).__init__()
        self.rfpad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(inC, 32, kernel_size=7, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.rfpad2 = nn.ReflectionPad2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0, bias=False)
        self.relu2 = nn.ReLU(inplace=True)   

        self.block1 = resBlock(64, 128, True, False)
        self.block2s = nn.ModuleList([resBlock(128, 128, False, False) for _ in range(6)])
        self.block3 = resBlock(128, 64, False, True)

        self.rfpad3 = nn.ReflectionPad2d(2)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=0, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.rfpad4 = nn.ReflectionPad2d(3)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0, bias=True)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.rfpad1(x)
        print("1", x.shape)
        x = self.relu1(self.conv1(x))    
        print("2", x.shape)
        x = self.rfpad2(x)
        print("3", x.shape)
        x = self.relu2(self.conv2(x))  
        print("4", x.shape)

        x = self.block1(x)
        print("5", x.shape)
        for i in range(6):
            x = self.block2s[i](x)
        print("6", x.shape)

        x = self.block3(x)
        print("7", x.shape)

        x = self.rfpad3(x)
        print("8", x.shape)
        x = self.relu3(self.conv3(x))     
        print("9", x.shape)

        x = self.rfpad4(x)
        print("10", x.shape)

        x = self.sigmoid1(self.conv4(x))   
        print("11", x.shape)
     
        return x


class resBlock(nn.Module):
    def __init__(self, inC, outC, downsample=False, upsample=False):
        super(resBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample

        self.rfpad1 = nn.ReflectionPad2d(1)
        if(self.downsample):
            self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=2, padding=0, bias=False)
        elif(self.upsample):
            self.conv1 = nn.ConvTranspose2d(inC, outC, kernel_size=3, stride=1, padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(outC)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.rfpad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(outC)
        self.relu2 = nn.ReLU(inplace=True)
        
        # residual for downsampling
        self.res_down = nn.Conv2d(inC, outC, kernel_size=1, stride=2, padding=0, bias=False)
        # residual for upsampling
        self.res_up1 = nn.Conv2d(inC, outC, kernel_size=1, stride=1, padding=0, bias=False)
        self.res_up2 = nn.Upsample(scale_factor=2, mode='nearest') 
    
    def forward(self, x):
        if(self.downsample):
            residual = self.res_down(x)
        elif(self.upsample):
            residual = self.res_up2(self.res_up1(x))
        else:
            residual = x
        x = self.rfpad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        print("down_sample1",x.shape)
        x = self.relu1(x) 

        x = self.rfpad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        print("down_sample2",x.shape)

        x += residual
        x = self.relu2(x)  
        return x
