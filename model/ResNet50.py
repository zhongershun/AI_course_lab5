import torch
import torch.nn as nn
import torchvision

# 残差块
class Bottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1,downsampling=False):
        super(Bottleneck,self).__init__()
        self.downsampling = downsampling

        ## 每个残差中有3个卷积层

        self.bottleneck = nn.Sequential(

            # 第一个卷积层：kernel_size = 1×1 ，padding = 0， stride = 1 
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1, bias=False),
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 第二个卷积层：kernel_size = 3×3 ，padding = 1， stride通常取2，但是只在该残差快的第一轮循环中会取到2，后续取默认值1，保证在第一轮经过残差快之后输出图片大小变为原来的1/2，后续保持不变 
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # 第三个卷积层：kernel_size = 1×1 ，padding = 0， stride = 1 
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels*4),
        )
        

        # 判断 x 的数据格式（维度）是否和 F(x)的一样，如果不一样，则进行一次卷积运算，实现升维操作。
        # 卷积核尺寸： 1×1 ，个数为原始特征图通道数的4倍， 填充值为 0， 步长为 1 
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
 
        residual = x        
        out = self.bottleneck(x)
        # print("x.shape",residual.shape)
        # print("out.shape",out.shape)
		## 下采样
        if self.downsampling:
            residual = self.downsample(x)
        
            # print("x.shape_after",residual.shape)
            # print("out.shape_after",out.shape)
        ## 结果和残差进行合并
        out += residual        
        out = self.relu(out)

        return out

class resnet(nn.Module):
    def Conv1(self,in_channels, out_channels, stride=2):
        conv1 = nn.Sequential(
            ## 属于的图像的大小为(224,224,3)
            ## 卷积核：kernel_size = 7×7 ，padding = 3， stride = 2, 默认64个卷积核
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=stride,padding=3,bias=False),
            ## 卷积后大小为(112,112,64)

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            ##池化层，kernel_size = 3×3 ，padding = 1， stride = 2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ## 经过这层后图像变为(56,56,64)，将作为stage1的输入
        )
        return conv1
    

    def make_layer(self, in_channels, out_channels, block, stride):
        layers = []

        ## 残差块的第一个输入的channels为in_channels
        layers.append(Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, downsampling = True))
        
        ## 残差快后续的输入都是in_channels的4倍
        ## 且后续残差快不需要下采样了
        for i in range(1, block):
            layers.append(Bottleneck(in_channels=out_channels*4,out_channels=out_channels))
        return nn.Sequential(*layers)


    def __init__(self,blocks, config):
        super(resnet, self).__init__()

        # 调用 stem 层 
        self.conv1 = self.Conv1(in_channels = 3, out_channels= 64)
		
		# stage 1-4
        self.layer1 = self.make_layer(in_channels = 64, out_channels = 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_channels = 256, out_channels = 128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_channels = 512, out_channels = 256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_channels = 1024, out_channels = 512, block=blocks[3], stride=2)
		
		# avgpool
        # self.avgpool = nn.AvgPool2d(7, stride=1)

        # 全连接层
        # self.fc = nn.Linear(2048,config.middle_hidden)
	    
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        self.hidden_state_transformer = nn.Sequential(
            # (7,7,2048)
            nn.Conv2d(in_channels=2048,out_channels=64,kernel_size=1),
            # (7,7,64)
            nn.Flatten(start_dim=2),
            # (64,7*7)
            nn.Dropout(0.2),
            nn.Linear(7 * 7, config.middle_hidden), 
            # (64,hidden_size)
            nn.ReLU(inplace=True)
        )
        self.trans = nn.Sequential(
            nn.AvgPool2d(7, stride=1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(2048, config.middle_hidden),
            nn.ReLU(inplace=True)
        )

        # 是否进行fine-tune
        for param in self.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True


    def forward(self, x):
        
        #(224,224,3)
        x = self.conv1(x)

        #(56,56,64)
        x = self.layer1(x)
        
        #(56,56,256)
        x = self.layer2(x)
        
        #(28,28,512)
        x = self.layer3(x)
        
        #(14,14,1024)
        x = self.layer4(x)
        
        hidden_state = self.hidden_state_transformer(x)

        #(7,7,2048)
        x = self.trans(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        #(num_class,)
        return hidden_state,x

        

if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = resnet([3,4,6,3],num_classes=64)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    hidden,feature = model(input)
    print(hidden.shape)
    print(feature.shape)
