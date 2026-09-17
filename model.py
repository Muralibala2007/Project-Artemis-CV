import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_lunar_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Adapt first conv layer for 1-channel grayscale
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1, 64, 
        kernel_size=original_conv.kernel_size, 
        stride=original_conv.stride, 
        padding=original_conv.padding, 
        bias=False
    )
    
    # Initialize the new 1-channel weights
    model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
    
    # Adapt the final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    return model