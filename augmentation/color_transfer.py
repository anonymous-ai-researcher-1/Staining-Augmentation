import torch
import torch.nn as nn

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, dim, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)

class StainEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, dim, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feat = self.encoder(x)
        stain_code = self.global_pool(feat).squeeze(-1).squeeze(-1)
        return stain_code

class Decoder(nn.Module):
    def __init__(self, content_dim=256, stain_dim=128, out_channels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(stain_dim, 16*16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(content_dim + 1, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, content_feat, stain_code):
        B, _, H, W = content_feat.shape
        stain_embed = self.fc(stain_code).view(B, 1, H, W)
        concat = torch.cat([content_feat, stain_embed], dim=1)
        return self.decoder(concat)
