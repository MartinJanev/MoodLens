import torch
import torch.nn as nn
import torch.nn.functional as F

def activation(act: str = "silu"):
    """
    Returns a PyTorch activation layer based on the given name.
    Supported activations: "relu", "elu", "silu" (default).
    ReLU - Rectified Linear Unit, is the most common activation. It outputs the input directly if it is positive; otherwise, it will output zero.
    ELU - Exponential Linear Unit, is similar to ReLU but has a smoother curve for negative inputs, which can help with learning.
    SiLU - Sigmoid Linear Unit (also known as Swish), is a smooth, non-monotonic activation function that has been shown to perform well in deep networks.
    If an unsupported activation is provided, it defaults to SiLU.
    """
    if act == "relu": return nn.ReLU(inplace=True)
    if act == "elu":  return nn.ELU(inplace=True)
    return nn.SiLU(inplace=True)  # default

class ConvBNAct(nn.Module):
    """
    A simple Conv2D + BatchNorm2d + Activation block.

    This is a common pattern in CNNs to normalize the output of convolutional layers
    and apply a non-linear activation function.

    :param c_in: Number of input channels.
    :param c_out: Number of output channels.
    :param k: Kernel size for the convolution (default is 3).
    :param s: Stride for the convolution (default is 1).
    :param p: Padding for the convolution (default is 1).
    :param act: Activation function to use (default is "silu").

    Supported activations are "relu", "elu", and "silu". If an unsupported
    activation is provided, it defaults to SiLU.
    :return: A callable module that applies the convolution, batch normalization,
             and activation in sequence.
    """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = activation(act)
    def forward(self, x):
        """
        Forward pass through the ConvBNAct block.
        :param x: Input tensor of shape (B, C_in, H, W).
        :return: Output tensor of shape (B, C_out, H', W'), where H' and W' are the height and width after convolution.
        The output tensor is the result of applying the convolution, batch normalization,
        """
        return self.act(self.bn(self.conv(x)))

class EmotionCNN(nn.Module):
    """
    Compact CNN for 48x48 grayscale FER with GAP head (no big FC),
    light SE, SiLU activations, and SpatialDropout2d regularization.
    Params drop massively; inference gets faster, especially on CPU.
    """
    def __init__(self, num_classes: int = 7, act: str = "silu"):
        super().__init__()
        self.stem   = ConvBNAct(1,   32, act=act)
        self.layer2 = nn.Sequential(ConvBNAct(32,  64, act=act), nn.MaxPool2d(2))        # 48→24
        self.layer3 = nn.Sequential(ConvBNAct(64, 128, act=act), ConvBNAct(128,128, act=act), nn.MaxPool2d(2))  # 24→12
        self.layer4 = nn.Sequential(ConvBNAct(128,256, act=act), nn.MaxPool2d(2))        # 12→6

        # SE-lite: use a tiny bottleneck MLP on channel avg
        self.se_fc1 = nn.Linear(256, 64)
        self.se_fc2 = nn.Linear(64, 256)

        # Regularize conv maps channel-wise
        self.spatial_drop = nn.Dropout2d(p=0.10)

        # Global average pool head (kills the huge FC)
        self.gap  = nn.AdaptiveAvgPool2d(1)     # (B,256,1,1)
        self.head = nn.Linear(256, num_classes) # ~1.8k params for 7 classes

    def forward(self, x):
        """
        Forward pass through the EmotionCNN model.
        :param x: Input tensor of shape (B, 1, 48, 48) where B is the batch size.
        :return: Output tensor of shape (B, num_classes) after applying the stem,
                    convolutional layers, SE-lite, spatial dropout, global average pooling,
                    and the final linear head.
        The output tensor contains the class logits for each input sample.
        """
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)      # (B,256,6,6)

        # SE-lite
        w = x.mean(dim=(2,3))   # (B,256)
        w = F.silu(self.se_fc1(w), inplace=True)
        w = torch.sigmoid(self.se_fc2(w)).unsqueeze(-1).unsqueeze(-1)  # (B,256,1,1)
        x = x * w

        x = self.spatial_drop(x)
        x = self.gap(x).flatten(1)  # (B,256)
        return self.head(x)
