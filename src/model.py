import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, finetune=True):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Add additional channel to the first convolutional layer
        conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(4, conv1.out_channels, 
                                        kernel_size=conv1.kernel_size, 
                                        stride=conv1.stride, 
                                        padding=conv1.padding, 
                                        bias=conv1.bias)
        
        # Copy pretrained weights for the first 3 channels (initialize new channel with R channel weights)
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = conv1.weight
            self.backbone.conv1.weight[:, 3] = conv1.weight[:, 0]
        
        # Replace FC layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

        # Freeze weights if necessary
        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Always train first conv layer and FC layer
        for param in self.backbone.conv1.parameters():
            param.requires_grad = True

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)