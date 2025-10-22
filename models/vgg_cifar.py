"""VGG models adapted for CIFAR-10.

Standard VGG architectures modified for 32x32 input images.
Neuron masking is applied to the first convolutional layer during training.
"""

import torch
import torch.nn as nn


# VGG configuration: number of filters in each conv layer
# 'M' denotes max pooling
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """VGG network for CIFAR-10.

    Architecture:
        - Convolutional features (based on VGG config)
        - Adaptive average pooling to 1x1
        - Classifier: Linear(512, 512) -> ReLU -> Linear(512, num_classes)
        - Neuron masking is applied to the first convolutional layer during training
    """

    def __init__(self, vgg_name, num_classes=10, use_batchnorm=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], use_batchnorm)

        # Adaptive pooling to handle different spatial sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier without dropout
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 32, 32)

        Returns:
            Logits of shape (batch, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, batch_norm):
        """Create convolutional feature layers.

        Args:
            cfg: List of layer configurations (int = filters, 'M' = maxpool)
            batch_norm: Whether to use batch normalization

        Returns:
            Sequential module containing conv layers
        """
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights using standard methods."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_masking_layer(self):
        """Get the layer where neuron masking should be applied.

        This is the ReLU activation after the first convolutional layer.
        In the features Sequential, this is index 2 (Conv2d, BatchNorm2d, ReLU).

        Returns:
            nn.Module: The ReLU layer where masking hooks should be attached
        """
        return self.features[2]  # ReLU after first Conv2d layer


def VGG11(num_classes=10, use_batchnorm=True):
    """VGG11 model for CIFAR-10."""
    return VGG('VGG11', num_classes, use_batchnorm)


def VGG13(num_classes=10, use_batchnorm=True):
    """VGG13 model for CIFAR-10."""
    return VGG('VGG13', num_classes, use_batchnorm)


def VGG16(num_classes=10, use_batchnorm=True):
    """VGG16 model for CIFAR-10."""
    return VGG('VGG16', num_classes, use_batchnorm)


def VGG19(num_classes=10, use_batchnorm=True):
    """VGG19 model for CIFAR-10."""
    return VGG('VGG19', num_classes, use_batchnorm)


# Test function
if __name__ == '__main__':
    # Test all models
    for model_fn, name in [(VGG11, 'VGG11'), (VGG13, 'VGG13'),
                            (VGG16, 'VGG16'), (VGG19, 'VGG19')]:
        model = model_fn()
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"{name}: input {x.shape} -> output {y.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Masking layer: {model.get_masking_layer()}")
