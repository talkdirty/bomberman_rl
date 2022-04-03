import torch
from torchvision.models.resnet import ResNet, BasicBlock

class CnnboardResNet(ResNet):
    """
    Standard ResNet implementation assumes images, so 3 input channels.
    We need to change this to accomodate our 5 channels.
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
                BasicBlock, 
                [2, 2, 2, 2], # ResNet-18
                6, # 6 output features in the fully connected layer
                norm_layer=lambda channels: torch.nn.GroupNorm(
                    num_groups=32, 
                    num_channels=channels, 
                    eps=1e-5, 
                    affine=True
                ),
                **kwargs,
        )
        # Override first convolutional layer to accept 5 channels
        self.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
