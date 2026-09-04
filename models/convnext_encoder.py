"""Compact optional PyTorch spatial encoder, no pretrained download required."""
try:
    import torch
    from torch import nn
    class CompactSpatialEncoder(nn.Module):
        def __init__(self, in_channels: int = 9, width: int = 24):
            super().__init__(); self.net = nn.Sequential(nn.Conv2d(in_channels,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU())
        def forward(self, x): return self.net(x)
except ImportError:
    class CompactSpatialEncoder:
        def __init__(self, *args, **kwargs): raise RuntimeError("PyTorch is optional; install torch to use neural training modules")
