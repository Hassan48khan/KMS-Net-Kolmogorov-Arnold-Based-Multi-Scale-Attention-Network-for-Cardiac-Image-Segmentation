import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

# ensure your kan_convolutional package path is on sys.path if needed
# sys.path.append('./kan_convolutional')  # uncomment if required in your environment

# Import the KANLinear primitive and convolution utilities
from KANLinear import KANLinear
import convolution  # expects multiple_convs_kan_conv2d implemented here


# ------------------------------------------------------------------
#             KAN: low-level KAN convolution wrapper classes
# ------------------------------------------------------------------

class KAN_Convolution(torch.nn.Module):
    """
    Single KAN convolution unit that maps a flattened kernel window to one output
    value using a KANLinear (spline-based) functional expansion.

    This unit expects the helper `convolution.multiple_convs_kan_conv2d` to
    know how to place this unit across the image and combine channels.
    """
    def __init__(
        self,
        kernel_size: tuple = (2, 2),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = [-1, 1],
        device = "cpu"
    ):
        super(KAN_Convolution, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.grid_size = grid_size
        self.spline_order = spline_order

        # KANLinear expects in_features = product(kernel_size) and returns 1 value
        self.conv = KANLinear(
            in_features = math.prod(kernel_size),
            out_features = 1,
            grid_size = grid_size,
            spline_order = spline_order,
            scale_noise = scale_noise,
            scale_base = scale_base,
            scale_spline = scale_spline,
            base_activation = base_activation,
            grid_eps = grid_eps,
            grid_range = grid_range
        )

    def forward(self, x: torch.Tensor):
        """
        Apply this KAN unit across the input using the helper in convolution.py.
        The helper handles extracting sliding windows, flattening, applying
        the KANLinear (self) and reassembling the output feature map.
        """
        # multiple_convs_kan_conv2d expects a list of KAN_Convolution-like
        # objects; passing [self] will compute a single-output conv (out_channels=1)
        return convolution.multiple_convs_kan_conv2d(
            x,
            [self],
            kernel_size=self.kernel_size[0],
            out_channels=1,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            device=x.device
        )

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        If the underlying KANLinear exposes a regularization loss (commonly).
        Fall back to zero if not present.
        """
        if hasattr(self.conv, "regularization_loss"):
            return self.conv.regularization_loss(regularize_activation, regularize_entropy)
        return torch.tensor(0., device=next(self.parameters()).device)


class KAN_Convolutional_Layer(torch.nn.Module):
    """
    KAN convolutional layer representing a full convolution between in_channels
    and out_channels. Internally constructs in_channels * out_channels KAN_Convolution
    modules and delegates the heavy-lifting to convolution.multiple_convs_kan_conv2d.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: tuple = (3,3),
        stride: tuple = (1,1),
        padding: tuple = (0,0),
        dilation: tuple = (1,1),
        grid_size: int = 5,
        spline_order:int = 3,
        scale_noise:float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = [-1, 1],
        device: str = "cpu"
    ):
        """
        Create a KAN convolutional layer by instancing one KAN_Convolution per
        input-output channel pair. This mirrors the parameterization of a
        conventional Conv2d but each "kernel parameter" is a KAN function.
        """
        super(KAN_Convolutional_Layer, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        # ModuleList of KAN_Convolution instances. There are in_channels * out_channels entries.
        self.convs = torch.nn.ModuleList()
        for _ in range(in_channels * out_channels):
            self.convs.append(
                KAN_Convolution(
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    dilation = dilation,
                    grid_size = grid_size,
                    spline_order = spline_order,
                    scale_noise = scale_noise,
                    scale_base = scale_base,
                    scale_spline = scale_spline,
                    base_activation = base_activation,
                    grid_eps = grid_eps,
                    grid_range = grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        """
        Apply the entire layer. The convolution.multiple_convs_kan_conv2d helper
        accepts the list of KAN_Convolution modules and assembles an output with
        shape (batch, out_channels, H_out, W_out).
        """
        self.device = x.device
        return convolution.multiple_convs_kan_conv2d(
            x,
            self.convs,
            kernel_size=self.kernel_size[0],
            out_channels=self.out_channels,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            device=self.device
        )

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Sum regularization losses from contained KAN units if available.
        """
        losses = []
        for k in self.convs:
            if hasattr(k, "regularization_loss"):
                losses.append(k.regularization_loss(regularize_activation, regularize_entropy))
        if len(losses) == 0:
            return torch.tensor(0., device=next(self.parameters()).device)
        return sum(losses)


# ------------------------------------------------------------------
#                 SE block and KASPPS module
# ------------------------------------------------------------------

class SEBlock(nn.Module):
    """Standard Squeeze-and-Excitation block (channel-wise recalibration)."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class KASPPS(nn.Module):
    """
    KASPPS: Atrous Spatial Pyramid Pooling using KAN convolutional layers and SE blocks.
    Replaces the 3x3 dilated Conv2d branches in standard ASPP with KAN-based branches.
    """
    def __init__(self, in_channels, out_channels):
        super(KASPPS, self).__init__()

        # KAN branches (dilations 6,12,18) with grid sizes 3,6,9 respectively
        self.kan6 = nn.Sequential(
            KAN_Convolutional_Layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                padding=(6,6),
                dilation=(6,6),
                grid_size=3
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

        self.kan12 = nn.Sequential(
            KAN_Convolutional_Layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                padding=(12,12),
                dilation=(12,12),
                grid_size=6
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

        self.kan18 = nn.Sequential(
            KAN_Convolutional_Layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                padding=(18,18),
                dilation=(18,18),
                grid_size=9
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

        # global pooling branch (kept as standard conv for projection)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

        # final fuse conv (1x1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=True)

        b6 = self.kan6(x)
        b12 = self.kan12(x)
        b18 = self.kan18(x)

        out = torch.cat([b6, b12, b18, gp], dim=1)
        out = self.fuse(out)
        return out


# ------------------------------------------------------------------
# If you want a quick test snippet (optional)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test (requires convolution.multiple_convs_kan_conv2d to be implemented)
    B, C_in, H, W = 2, 3, 64, 64
    x = torch.randn(B, C_in, H, W)
    model = KASPPS(in_channels=C_in, out_channels=64)
    try:
        y = model(x)
        print("KASPPS output shape:", y.shape)
    except Exception as e:
        print("Smoke test failed (this is expected if helper functions are not implemented):", e)
