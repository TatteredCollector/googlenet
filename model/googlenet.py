import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


# 结一下，两个库都可以实现神经网络的各层运算。
# 包括卷积、池化、padding、激活(非线性层)、
# 线性层、正则化层、其他损失函数Loss，
# 两者都可以实现不过nn.functional毕竟只是nn的子库，
# nn的功能要多一些，还可以实现如Sequential()这种将多个层弄到一个序列这样复杂的操作。
#


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class inception(nn.Module):
    def __init__(self, in_channel, ch1_out, ch3red, ch3, ch5red, ch5, proj):
        super(inception, self).__init__()
        self.branch_1 = BasicConv(in_channel=in_channel,
                                  out_channel=ch1_out, kernel_size=1)
        self.branch_2 = nn.Sequential(BasicConv(in_channel=in_channel,
                                                out_channel=ch3red, kernel_size=1),
                                      BasicConv(in_channel=ch3red,
                                                out_channel=ch3, kernel_size=3,
                                                padding=1)
                                      )
        self.branch_3 = nn.Sequential(BasicConv(in_channel=in_channel, out_channel=ch5red,
                                                kernel_size=1),
                                      BasicConv(in_channel=ch5red,
                                                out_channel=ch5, kernel_size=5,
                                                padding=2))
        self.branch_4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                      BasicConv(in_channel=in_channel,
                                                out_channel=proj,
                                                kernel_size=1))

    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(x)
        branch3 = self.branch_3(x)
        branch4 = self.branch_4(x)
        # 用torch.cat((A,B),dim)时，
        # 除拼接维数dim数值可不同外其余维数数值需相同，方能对齐。
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class inception_aux(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(inception_aux, self).__init__()
        '''self.aux = nn.Sequential(nn.AdaptiveAvgPool2d(kernel_size=5,
                                                      stride=3),
                                 BasicConv(in_channel=in_channel, out_channel=128,
                                           kernel_size=1,
                                           stride=1),
                                 nn.Flatten(),
                                 nn.Dropout(p=0.4),
                                 nn.Linear(in_features=4*4*128, out_features=1024),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.4),
                                 nn.Linear(in_features=1024, out_features=num_classes)

                                 )
        # 调用nn 进行实现
        '''
        # 调用函数形式实现
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv(in_channel=in_channel, out_channel=128,
                              kernel_size=1, stride=1)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        # aux1 : 512*14*14->512*4*4  aux2:528*14*14->528*4*4
        x = self.conv(x)
        # aux1: 512*4*4 aux2:528*4*4->128*4*4
        x = torch.flatten(x, 1)
        #  128*4*4->2048
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        #  2048->1024
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # 1024->num_classes
        return x


class googleNet(nn.Module):
    def __init__(self, AuxFlage=True, num_classes=1000, init_weight=False):
        super(googleNet, self).__init__()
        self.aux_flage = AuxFlage
        self.conv1 = BasicConv(in_channel=3, out_channel=64,
                               kernel_size=7, stride=2, padding=3)
        # ceil_mode - 如果等于True，
        # 计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv(in_channel=64, out_channel=64,
                               kernel_size=1, stride=1)
        self.conv3 = BasicConv(in_channel=64, out_channel=192,
                               kernel_size=3, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception_3a = inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception_4a = inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception_5a = inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = inception(832, 384, 192, 384, 48, 128, 128)
        # 先只知道的是输入数据和输出数据的大小，
        # 而不知道核与步长的大小。如果使用上面的方法创建汇聚层，
        # 我们每次都需要手动计算核的大小和步长的值。
        # 而自适应（Adaptive）能让我们从这样的计算当中解脱出来，
        # 只要我们给定输入数据和输出数据的大小，
        # 自适应算法能够自动帮助我们计算核的大小和每次移动的步长。
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.liner = nn.Linear(in_features=1024, out_features=num_classes)
        if self.aux_flage:
            self.aux_1 = inception_aux(in_channel=512, num_classes=num_classes)
            self.aux_2 = inception_aux(in_channel=528, num_classes=num_classes)
        if init_weight:
            self._init_weight()

    def forward(self, x):
        # N*3*224*224->N*64*112*112
        x = self.conv1(x)
        # N*3*112*112->N*64*56*56
        x = self.max_pool_1(x)
        # N*64*56*56->N*64*56*56
        x = self.conv2(x)
        # N*64*56*56->N*192*56*56
        x = self.conv3(x)
        # N*192*56*56->N*192*28*28
        x = self.max_pool_2(x)

        # N*192*28*28->N*256*28*28
        x = self.inception_3a(x)
        # N*256*28*28 -> N*480*28*28
        x = self.inception_3b(x)
        # N*480*28*28->N*480*14*14
        x = self.max_pool_3(x)

        # N*480*14*14 ->N*512*14*14
        x = self.inception_4a(x)
        if self.training and self.aux_flage:
            aux_x1 = self.aux_1(x)
        # N*512*14*14 -> N*512*14*14
        x = self.inception_4b(x)
        # N*512*14*14 -> N*512*14*14
        x = self.inception_4c(x)
        # N*512*14*14 -> N*528*14*14
        x = self.inception_4d(x)
        if self.training and self.aux_flage:
            aux_x2 = self.aux_2(x)
        # N*528*14*14 -> N*832*14*14
        x = self.inception_4e(x)
        # N*832*14*14 -> N*832*7*7
        x = self.max_pool_4(x)
        # N*832*7*7 -> N*832*7*7
        x = self.inception_5a(x)
        # N*832*7*7 ->N*1024*7*7
        x = self.inception_5b(x)
        # N*1027*7*7 -> N*1024*1*1
        x = self.avg_pool(x)
        # N*1024*1*1 ->N*1024
        x = torch.flatten(x, start_dim=1)
        x = torch.dropout(x, p=0.4, train=self.training)
        # N*1024 -> N*num_classes
        x = self.liner(x)
        if self.training and self.aux_flage:
            return x, aux_x1, aux_x2
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)




if __name__ == "__main__":
    net = googleNet(init_weight=True)
    # torch 网络输入格式是 BN， channel , W , H
    # inputs = torch.randn(16, 3, 224, 224)
    # out = net(inputs)
    # 使用torchsummary.summary()函数 模型默认构建在gpu上
    # summary(your_model, input_size=(channels, H, W))
    net.to(torch.device("cuda:0"))
    summary(net, input_size=(3, 224, 224))
