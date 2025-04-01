
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# Get the initial number of channels for each stage
def get_inplanes():
    return [64, 128, 256, 512]

# Define a 3x3x3 convolution operation
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

# Define a 1x1x1 convolution operation
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

# Define the Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Define the Bottleneck Block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Define the Cross-Attention Mechanism
class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.query_conv_A = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv_A = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv_A = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.query_conv_B = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv_B = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv_B = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.gamma_A = nn.Parameter(torch.zeros(1))
        self.gamma_B = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, D, H, W = x.size()

        # Task A
        Q_taskA = self.query_conv_A(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        K_taskA = self.key_conv_A(x).view(batch_size, -1, D * H * W)
        V_taskA = self.value_conv_A(x).view(batch_size, -1, D * H * W)

        # Task B
        Q_taskB = self.query_conv_B(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        K_taskB = self.key_conv_B(x).view(batch_size, -1, D * H * W)
        V_taskB = self.value_conv_B(x).view(batch_size, -1, D * H * W)

        # Cross-attention: Task B attends to Task A
        attention_B_A = self.softmax(torch.bmm(Q_taskB, K_taskA))
        F_B_new = torch.bmm(V_taskA, attention_B_A.permute(0, 2, 1))
        F_B_new = F_B_new.view(batch_size, C, D, H, W)

        # Cross-attention: Task A attends to Task B
        attention_A_B = self.softmax(torch.bmm(Q_taskA, K_taskB))
        F_A_new = torch.bmm(V_taskB, attention_A_B.permute(0, 2, 1))
        F_A_new = F_A_new.view(batch_size, C, D, H, W)

        # Apply the learned gamma coefficients
        out_task1 = self.gamma_A * F_A_new + x
        out_task2 = self.gamma_B * F_B_new + x

        return out_task1, out_task2

#  ResNet-CA Model
class ResNetWithAttention(nn.Module):
    def __init__(self, block, layers, block_inplanes, n_input_channels=1, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes_task1=2, n_classes_task2=2):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(conv1_t_size, 7, 7), stride=(conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.cross_attention = CrossAttention(block_inplanes[3] * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 平均池化移到交叉注意力之后

        self.fc_task1 = nn.Linear(block_inplanes[3] * block.expansion, n_classes_task1)
        self.fc_task2 = nn.Linear(block_inplanes[3] * block.expansion + n_classes_task1, n_classes_task2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out_task1, out_task2 = self.cross_attention(x)
        out_task1 = self.avgpool(out_task1).view(out_task1.size(0), -1)
        out_task2 = self.avgpool(out_task2).view(out_task2.size(0), -1)
        out_task1 = self.fc_task1(out_task1)
        out_task2 = torch.cat((out_task2, out_task1), dim=1)
        out_task2 = self.fc_task2(out_task2)

        return out_task1, out_task2

# Generate the model based on the model depth
def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNetWithAttention(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNetWithAttention(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNetWithAttention(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNetWithAttention(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNetWithAttention(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNetWithAttention(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNetWithAttention(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
