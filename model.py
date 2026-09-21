import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
)

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.num_features = num_features

    def forward(self, x, cond):
        gamma_beta = self.to_gamma_beta(cond)
        gamma = gamma_beta[:, :self.num_features].unsqueeze(-1).unsqueeze(-1)
        beta = gamma_beta[:, self.num_features:].unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights, 128),
    "resnet34": (resnet34, ResNet34_Weights, 128),
    "resnet50": (resnet50, ResNet50_Weights, 512), 
}

class LunarModel(nn.Module):
    def __init__(self, backbone_name="resnet34", pretrained=True, dropout=0.3, in_channels=3):
        super().__init__()
        if backbone_name not in _BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        ctor, weights_cls, film_channels = _BACKBONES[backbone_name]

        weights = weights_cls.DEFAULT if pretrained else None
        net = ctor(weights=weights)

        original_conv = net.conv1
        new_conv = nn.Conv2d(
            in_channels, original_conv.out_channels,
            kernel_size=original_conv.kernel_size, stride=original_conv.stride,
            padding=original_conv.padding, bias=False,
        )
        if pretrained:
            avg_w = original_conv.weight.data.mean(dim=1, keepdim=True)  
            new_conv.weight.data[:, 0:1, :, :] = avg_w
        net.conv1 = new_conv

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool
        self.feat_dim = net.fc.in_features

        self.film = FiLM(cond_dim=2, num_features=film_channels)

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, image, azimuth_feat):
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.film(x, azimuth_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        combined = torch.cat([x, azimuth_feat], dim=1)
        return self.classifier(combined)

def get_lunar_model(backbone_name: str = "resnet34", pretrained: bool = True, in_channels: int = 3):
    return LunarModel(backbone_name=backbone_name, pretrained=pretrained, in_channels=in_channels)