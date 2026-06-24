import os
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# Placeholder weights path comment as requested:
# WEIGHTS_PATH = "models/csrnet_shanghaitech.pth"

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    Constructs a sequential series of convolutional/pooling layers based on a config list.
    
    Args:
        cfg (list): List containing integers (channel depth) or 'M' (MaxPool2d).
        in_channels (int): Input channel count for the first layer (typically 3 for RGB).
        batch_norm (bool): If True, inserts a BatchNorm2d layer after each Conv2d layer.
        dilation (bool): If True, uses a dilation rate of 2 (Configuration B) and padding of 2
                         to preserve spatial size. If False, standard dilation=1, padding=1 is used.
                         
    Returns:
        nn.Sequential: PyTorch sequential module containing the constructed layers.
    """
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Receptive field expands using dilation while padding preserves image boundaries.
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """
    CSRNet Architecture (Dilated Convolutional Neural Networks for Understanding Highly Congested Scenes).
    
    Consists of:
      1. Front-end: VGG-16 first 10 convolutional layers for feature extraction (pretrained on ImageNet).
      2. Back-end: Dilated CNN (Configuration B - dilation rate 2) to capture density patterns without pooling.
      3. Output: 1x1 Convolution to generate the final single-channel density map.
    """
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # Configuration list for front-end (first 10 conv layers of VGG-16)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        
        # Configuration list for back-end (6 dilated convolutional layers)
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        # Build front-end features extraction and back-end dilated convolution layers
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        
        # Final output mapping from 64 features to a single density value per pixel region
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Apply standard initialization
        self._initialize_weights()
        
        # If load_weights is False, we initialize the front-end with pretrained VGG-16 weights
        if not load_weights:
            try:
                # Load standard pretrained VGG16 features from torchvision
                print("CSRNet: Fetching/Loading pretrained VGG-16 weights for feature extraction...")
                vgg = vgg16(weights=VGG16_Weights.DEFAULT)
                
                # Align conv layers from PyTorch VGG-16 features to our custom self.frontend
                vgg_convs = [layer for layer in vgg.features if isinstance(layer, nn.Conv2d)]
                frontend_convs = [layer for layer in self.frontend if isinstance(layer, nn.Conv2d)]
                
                # Copy weight and bias parameter tensors
                for target_conv, src_conv in zip(frontend_convs, vgg_convs[:len(frontend_convs)]):
                    target_conv.weight.data.copy_(src_conv.weight.data)
                    if target_conv.bias is not None and src_conv.bias is not None:
                        target_conv.bias.data.copy_(src_conv.bias.data)
                print("CSRNet: Successfully loaded pretrained VGG-16 weights into front-end.")
            except Exception as e:
                print(f"CSRNet: Warning - Could not load pretrained VGG-16 weights: {e}.")
                print("CSRNet: Frontend initialized with random/initialized weights.")

    def forward(self, x):
        """
        Executes a forward pass on the input tensor.
        
        Args:
            x (torch.Tensor): Input batch of frames. Shape: (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Predicted density map. Shape: (batch_size, 1, height/8, width/8)
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        """
        Manually initializes convolution and batchnorm layers.
        Convolutions are initialized using a normal distribution (std=0.01).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def load_csrnet_model(weights_path, device):
    """
    Loads pretrained weights from a saved .pth file and returns the CSRNet model in evaluation mode.
    
    Args:
        weights_path (str): Filepath pointing to the saved model checkpoint (.pth).
        device (torch.device): Device to load tensors onto (e.g. cpu or cuda).
        
    Returns:
        CSRNet: CSRNet model configured for inference on the selected device.
    """
    # Initialize CSRNet with load_weights=True to skip automatic download of VGG-16 weights
    model = CSRNet(load_weights=True)
    
    # Load state dict
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle checkpoints saved either as a raw state dict or nested under a key
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model
