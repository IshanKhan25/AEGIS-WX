try:
    import torch
    from torch import nn
    class TemporalUNet(nn.Module):
        def __init__(self, channels: int = 24):
            super().__init__(); self.net=nn.Sequential(nn.Conv3d(channels,channels,3,padding=1),nn.GELU(),nn.Conv3d(channels,channels,3,padding=1),nn.GELU())
        def forward(self,x): return self.net(x)
except ImportError:
    class TemporalUNet:
        def __init__(self,*args,**kwargs): raise RuntimeError("PyTorch required for TemporalUNet")
