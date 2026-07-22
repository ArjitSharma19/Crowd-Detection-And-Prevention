import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResCSRNet(nn.Module):
    """
    ResCSRNet Architecture: ResNet-50 Frontend + Dilated CNN Backend.
    
    Consists of:
      1. Front-end: Pretrained ResNet-50 features up through layer2 (stride 8, 512 channels).
      2. Back-end: Dilated CNN (dilation rate 2) to capture multi-scale crowd density patterns.
      3. Output: 1x1 Convolution to generate the final single-channel density map (stride 8).
    """
    def __init__(self, load_weights=False):
        super(ResCSRNet, self).__init__()
        self.seen = 0
        
        # 1. Front-end: ResNet-50 backbone
        res50 = resnet50(weights=ResNet50_Weights.DEFAULT if not load_weights else None)
        
        # Stem and initial layers up through layer2 (downsample by factor of 8)
        self.frontend = nn.Sequential(
            res50.conv1,
            res50.bn1,
            res50.relu,
            res50.maxpool, # Stride 4
            res50.layer1,  # Stride 4, 256 output channels
            res50.layer2   # Stride 8, 512 output channels
        )
        
        # 2. Back-end: 6 dilated convolution layers (in_channels=512)
        backend_cfg = [512, 512, 512, 256, 128, 64]
        layers = []
        in_channels = 512
        for v in backend_cfg:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True)
            ]
            in_channels = v
        self.backend = nn.Sequential(*layers)
        
        # 3. Output layer: 1x1 Conv mapping 64 channels to 1 density map channel
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Initialize weights for backend and output layer
        self._initialize_backend_weights()

    def forward(self, x):
        """
        Executes forward pass.
        Input x shape: (B, 3, H, W)
        Output density map shape: (B, 1, H/8, W/8)
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_backend_weights(self):
        """
        Initializes backend convolutional layers with Gaussian distribution (std=0.01).
        """
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)


def load_rescsrnet_model(weights_path, device):
    """
    Loads pretrained weights from a saved .pth file into ResCSRNet.
    
    Args:
        weights_path (str): Path to checkpoint file.
        device (torch.device): CUDA or CPU device target.
        
    Returns:
        ResCSRNet: Model initialized and set to eval mode.
    """
    model = ResCSRNet(load_weights=True)
    checkpoint = torch.load(weights_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
