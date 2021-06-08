import torch.nn as nn
from .utils import load_state_dict_from_url
import models.ReSprop_conv as C
iteration_size = 0
batch_size = 0
start_epoch = 0
sparsity = 0
warmup = 0


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, layer_number, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return  C.ReSConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, dilation = dilation, groups = groups, bias=True, layer=layer_number, iteration=iteration_size, batch=batch_size, epoch=start_epoch, sparse=sparsity, warm=warmup)

    
    #return C.MyConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    #  padding=1, dilation = dilation ,groups = groups, bias=False,name=layer_num , number=391 ,batch=1, Epoch=1 )


def conv1x1(in_planes, out_planes, layer_number, stride=1):
    """1x1 convolution"""
    
    
    return  C.ReSConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1, bias=True, layer=layer_number, iteration=iteration_size, batch=batch_size, epoch=start_epoch, sparse=sparsity, warm=warmup)

    #return C.MyConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, name=layer_num, number=391, batch=1 , Epoch=1)


class BasicBlock(nn.Module):
    expansion = 1
     
    def __init__(self, inplanes, planes, layer_number = [-1], stride=1, downsample=None, groups=1,
                 base_width=64,  norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        layer_number[0]+=1
        #print("basi_black", layer_number)
        self.conv1 = conv3x3(inplanes, planes, layer_number[0], stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        layer_number[0]+=1
        #print("basi_black", layer_number)
        self.conv2 = conv3x3(planes, planes,layer_number[0])
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer_number = -1, stride=1, downsample=None, groups=1,
                 base_width=64,  norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        layer_number[0]+=1
        self.conv1 = conv1x1(inplanes, width,layer_number[1])
        self.bn1 = norm_layer(width)
        layer_number[0]+=1
        self.conv2 = conv3x3(width, width, layer_number[1],stride, groups, dilation)
        self.bn2 = norm_layer(width)
        layer_number[0]+=1
        self.conv3 = conv1x1(width, planes * self.expansion,layer_number[1])
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    layer_number = [1] 
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, 
                norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.layer_number[0] = 0 
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.layer_number[0]+=1
        self.conv1 = C.ReSConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, layer=self.layer_number[0], iteration=iteration_size, batch=batch_size, epoch=start_epoch, sparse=sparsity, warm=warmup)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], layer_no=self.layer_number, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1],layer_no=self.layer_number, stride=2,
                                       norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], layer_no=self.layer_number, stride=2,
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3],layer_no=self.layer_number,stride=2,
                                       norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, C.ReSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, layer_no,stride=1, norm_layer=None):
        if norm_layer is None:
          norm_layer = nn.BatchNorm2d
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            self.layer_number[0]+=1
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, self.layer_number[0], stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, layer_no, stride, downsample, self.groups,
                            self.base_width,  norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,layer_no, groups=self.groups,
                                base_width=self.base_width,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(iteration, batch, epoch, sparse, warm, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    global iteration_size
    global batch_size
    global start_epoch
    global sparsity
    global warmup

    iteration_size = iteration
    batch_size = batch
    start_epoch = epoch
    sparsity = sparse
    warmup = warm



    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(iteration, batch, epoch, sparse, warm, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    global iteration_size
    global batch_size
    global start_epoch
    global sparsity
    global warmup

    iteration_size = iteration
    batch_size = batch
    start_epoch = epoch
    sparsity = sparse
    warmup = warm




    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(iteration, batch, epoch, sparse, warm, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    global iteration_size
    global batch_size
    global start_epoch
    global sparsity
    global warmup

    iteration_size = iteration
    batch_size = batch
    start_epoch = epoch
    sparsity = sparse
    warmup = warm


    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
