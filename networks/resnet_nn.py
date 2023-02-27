from turtle import xcor
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

TEACHER_NET_NAME = {4: 'resnet34',
                    5: 'resnet42'}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                inplanes,
                planes,
                stride=1,
                downsample=None,
                groups=1,
                base_width=64,
                dilation=1,
                norm_layer=None,
                teacher_outplanes=None
                ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d 
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        if teacher_outplanes is not None:
            out_planes = teacher_outplanes
        else:
            out_planes = planes
        self.conv2 = conv3x3(planes, out_planes)
        self.bn2 = norm_layer(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, params=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # print(identity.shape, x.shape, out.shape)
            identity = self.downsample(x) #, params=self.get_subdict(params, 'downsample'))
        try:
            out += identity
        except:
            print('out+iden', out.shape, identity.shape)
            import pdb; pdb.set_trace()
        out = self.relu(out)
        return out
    
    
class IdentityShortCut(nn.Module):
    def __init__(self, inplane, outplane, stride):
        super().__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.stride = stride
        
    def down_sampling(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.outplane-self.inplane))
        if self.stride == 2:
            out = nn.MaxPool2d(2, stride=self.stride)(out)
        return out
    
    def forward(self, x):
        out = x
        if self.stride != 1 or self.inplane != self.outplane:
            out = self.down_sampling(x)
        return out


# class IdentityShortCut(nn.Module):
#     def __init__(self, inplane, outplane, stride):
#         super().__init__()
#         self.inplane = inplane
#         self.outplane = outplane
#         self.stride = stride
#         self.s = None
#         self.r = None
#         if inplane > outplane:
#             self.s = int(inplane / outplane)
#         else:
#             self.r = int(outplane / inplane)
    
        
#     def forward(self, x, params=None):
#         if self.s:
#             # print('plane', self.inplane, self.outplane)
#             out = x[:, ::self.s, ::self.stride, ::self.stride]
#             if out.size(1) > self.outplane:
#                 out = out[:, :self.outplane]
#             return out
#         else:
#             # print('plane2', self.inplane, self.outplane, self.r)
#             out = x[:, :, ::self.stride, ::self.stride].repeat(1, self.r, 1, 1)
            
#             if out.size(1) < self.outplane:
#                 out = x[:, :, ::self.stride, ::self.stride].repeat(1, self.r+1, 1, 1)
#                 out = out[:, :self.outplane]
#             return out
        

class ResNetSmall(nn.Module):
    def __init__(self,
                block,
                depth_config,
                num_classes=1000,
                zero_init_residual=False,
                groups=1,
                width_per_group=64,
                replace_stride_with_dilation=None,
                norm_layer=None,
                channel_widths=[[32, 32, 32, 32, 32], [64, 64, 64, 64, 64], [128, 128, 128, 128, 128], [256, 256, 256, 256, 256]],
                stage_strides=[1, 2, 2, 2],
                tc_stage_cws=[32, 64, 128, 256]
                ):
        super(ResNetSmall, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d 
        self._norm_layer = norm_layer

        self.inplanes = channel_widths[0][0] # tc_stage_cws[0]
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, channel_widths[0], depth_config[0], stride=stage_strides[0]) 
        self.layer2 = self._make_layer(block, channel_widths[1], depth_config[1], stride=stage_strides[1])  
        self.layer3 = self._make_layer(block, channel_widths[2], depth_config[2], stride=stage_strides[2])
        self.layer4 = self._make_layer(block, channel_widths[3], depth_config[3], stride=stage_strides[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_widths[-1][depth_config[-1]-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes[0]:
            downsample = IdentityShortCut(self.inplanes, planes[0] * block.expansion, stride) 
        else:
            downsample = None
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes[0],
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )

        self.inplanes = planes[0] * block.expansion
        for i in range(1, blocks):
            if self.inplanes != planes[i]:
                downsample = IdentityShortCut(self.inplanes, planes[i] * block.expansion, stride=1) 
            else:
                downsample = None
            layers.append(
                block(
                    self.inplanes,
                    planes[i],
                    downsample=downsample,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )
            self.inplanes = planes[i]
        return nn.Sequential(*layers)

    def _forward_impl(self, x, get_features=False):

        if get_features: features = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if get_features:
            features[0] = x.detach()

        x = self.layer1(x)
        if get_features:
            features[1] = x.detach()
        x = self.layer2(x)
        if get_features:
            features[2] = x.detach()
        x = self.layer3(x)
        if get_features:
            features[3] = x.detach()
        x = self.layer4(x)
        if get_features:
            features[4] = x.detach()

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if get_features:
            return features, x.detach()
        return x

    def forward(self, x, get_features=False):
        return self._forward_impl(x, get_features=get_features)


def get_resnet_stage(stage, ks, depth_config, channel_width, stage_strides, tc_stage_cws):
    resnet =  ResNetSmall(BasicBlock,
                        depth_config=depth_config,
                        channel_widths=channel_width,
                        stage_strides=stage_strides,
                        tc_stage_cws=tc_stage_cws)
    if stage == 0:
        return resnet.layer1
    elif stage == 1:
        return resnet.layer2
    elif stage == 2:
        return resnet.layer3
    elif stage == 3:
        return resnet.layer4


def get_resnet(num_classes, depth_config, channel_widths, stage_strides, tc_stage_cws):
    return ResNetSmall(BasicBlock, 
                        num_classes=num_classes, 
                        depth_config=depth_config, 
                        channel_widths=channel_widths,
                        stage_strides=stage_strides,    
                        tc_stage_cws=tc_stage_cws)


def resnet18_small(num_classes):
    return ResNetSmall(BasicBlock, depth_config=[2, 2, 2, 2], num_classes=num_classes)


def resnet34_small(num_classes):
    return ResNetSmall(BasicBlock, depth_config=[4, 4, 4, 4], num_classes=num_classes)


def resnet42_small(num_classes):
    return ResNetSmall(BasicBlock, depth_config=[5, 5, 5, 5], num_classes=num_classes)


if __name__ == "__main__":
    tc_net = resnet42_small(num_classes=10)
    net = get_resnet(num_classes=10, 
                    depth_config=[2,1,3,4], 
                    channel_widths=[[16,16,16,16,16],[32,32,32,32,32],[64,64,64,64,64],[64,64,64,64,64]], 
                    stage_strides=[1,2,2,2],
                    tc_stage_cws=[32, 64, 128, 256])
    inp = torch.randn((1, 3, 64, 64))
    net = net.cuda()
    inp = inp.cuda()
    state_dict = net.state_dict()
    out = net(inp, state_dict)
    breakpoint()