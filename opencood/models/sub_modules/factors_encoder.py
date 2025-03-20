import torch as th
import torch.nn as nn
import torch.nn.functional as F

class EncResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(EncResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class Encoder4(nn.Module):
    def __init__(self, d, context_dim, latent_unit, bn=True, num_channels=3):
        super(Encoder4, self).__init__()
        self.context_dim = context_dim
        self.latent_unit = latent_unit
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            EncResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            EncResBlock(d, d, bn=bn),
            View((-1, 128 * 4 * 4)),                  
            nn.Linear(2048, self.latent_unit)               # batch_size x 160 -> B x 10 x 16
        )
        self.net = nn.ModuleList()
        for i in range(self.latent_unit):
            self.net.append(nn.Sequential(
                nn.Linear(1,64),
                nn.ELU(True),
                nn.Linear(64,128),
                nn.ELU(True),
                nn.Linear(128,self.context_dim),
            ))
        
    def forward(self,x):
        x = self.encoder(x)
        # print(x.shape)
        out = []
        for i in range(self.latent_unit):
            out.append(self.net[i](x[:,i][:,None]))
        return th.cat(out, dim=1)


    def encoding(self,x):
        return self.encoder(x)

    def warp(self,x):
        out = []
        for i in range(self.latent_unit):
            out.append(self.net[i](x[:,i][:,None]))
        return th.cat(out, dim=1)
    
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)    