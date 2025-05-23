"""
Sourced from
https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10
==============================================================================
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
from pylot.torch.torchlib import torch, nn, F


__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def add_early_maxpool(resnet):
    assert isinstance(resnet, ResNet)
    resnet.layer1 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet.layer1
    )
    return resnet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, early_maxpool=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)
        self.apply(_weights_init)

        self.fc.is_classifier = True  # So layer is not pruned
        default_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = default_maxpool if early_maxpool else nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet_factory(filters, num_classes, weight_file):
    def _resnet(pretrained=False, **kwargs):
        model = ResNet(BasicBlock, filters, num_classes=num_classes, **kwargs)
        if pretrained:
            # weights = weights_path(weight_file)
            # weights = torch.load(weights)["state_dict"]
            # # TODO have a better solution for DataParallel models
            # # For models trained with nn.DataParallel
            # if list(weights.keys())[0].startswith("module."):
            #     weights = {k[len("module."):]: v for k, v in weights.items()}
            # model.load_state_dict(weights)

            state_dict = torch.hub.load_state_dict_from_url(
                f"https://github.com/JJGO/shrinkbench-models/raw/master/cifar10/{weight_file}"
            )["state_dict"]

            for k in list(state_dict.keys()):
                if k.startswith("module."):
                    k2 = k[len("module.") :]
                    state_dict[k2] = state_dict.pop(k)
                    k = k2
                if k.startswith("linear."):
                    # Changed convention to simplify replace_head
                    k2 = k.replace("linear.", "fc.")
                    state_dict[k2] = state_dict.pop(k)

            model.load_state_dict(state_dict)
        return model

    return _resnet


resnet20 = resnet_factory([3, 3, 3], 10, "resnet20.th")
resnet32 = resnet_factory([5, 5, 5], 10, "resnet32.th")
resnet44 = resnet_factory([7, 7, 7], 10, "resnet44.th")
resnet56 = resnet_factory([9, 9, 9], 10, "resnet56.th")
resnet110 = resnet_factory([18, 18, 18], 10, "resnet110.th")
resnet1202 = resnet_factory([200, 200, 200], 10, "resnet1202.th")
