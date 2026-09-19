import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class LunarModel(nn.Module):
    """
    ResNet backbone with a 2-channel input:
      channel 0: normalized grayscale image
      channel 1: sun-aligned directional-gradient map (see dataset.py)
    plus a late-fusion branch that concatenates sin/cos(sun_azimuth_angle)
    onto the pooled feature vector before the final classifier.
    """
    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        elif backbone_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        original_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if pretrained:
            # channel 0 (grayscale): reuse averaged pretrained RGB weights
            new_conv.weight.data[:, 0:1, :, :] = original_conv.weight.data.mean(dim=1, keepdim=True)
            # channel 1 (gradient map): left at its default random init --
            # the network discovers how to use it during fine-tuning rather
            # than starting from a borrowed (meaningless, for this channel)
            # pretrained weight.
        backbone.conv1 = new_conv

        self.feat_dim = backbone.fc.in_features  # 512 for resnet18/34
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, image, azimuth_feat):
        feats = self.backbone(image)                        # (B, feat_dim)
        combined = torch.cat([feats, azimuth_feat], dim=1)   # (B, feat_dim + 2)
        return self.classifier(combined)


def get_lunar_model(backbone_name: str = "resnet34", pretrained: bool = True):
    return LunarModel(backbone_name=backbone_name, pretrained=pretrained)
