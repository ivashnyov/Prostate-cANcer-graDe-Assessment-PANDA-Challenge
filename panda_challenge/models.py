import timm
import torch.nn as nn
import torch
import torch.nn.functional as F


class ClassifcationMultiCropModel(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=6, **kwargs):
        super().__init__()
        m = timm.create_model(
            model_name,
            **kwargs)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(nc, num_classes))

    def forward(self, x):
        shape = x[0].shape
        n = len(x[0])
        x = x.view(-1, shape[1], shape[2], shape[3])
        # x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        # should be: bs*N x C x 4 x 4
        shape = x.shape
        # concatenate the output for tiles into a single map
        x = x.view(-1, n, shape[1], shape[2], shape[3])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, shape[1], shape[2]*n, shape[3])
        # should be: bs x C x N*4 x 4
        x = self.head(x)
        return x


class ClassifcationMultiCropModelMultiHead(nn.Module):
    def __init__(
        self,
        model_name='resnet34',
        num_classes_isup=6,
        num_gleason_major=4,
        num_gleason_minor=4,
            **kwargs):
        super().__init__()
        m = timm.create_model(
            model_name,
            **kwargs)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.features = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.ReLU(),
            nn.Flatten())
        self.head_isup = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(nc, nc//4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(nc//4, num_classes_isup),
            )
        self.head_gleason_major = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(nc, nc//4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(nc//4, num_gleason_major),
            )
        self.head_gleason_minor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(nc, nc//4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(nc//4, num_gleason_minor),
            )

    def forward(self, x):
        shape = x[0].shape
        n = len(x[0])
        x = x.view(-1, shape[1], shape[2], shape[3])
        # x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        # should be: bs*N x C x 4 x 4
        shape = x.shape
        # concatenate the output for tiles into a single map
        x = x.view(-1, n, shape[1], shape[2], shape[3])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, shape[1], shape[2]*n, shape[3])
        # should be: bs x C x N*4 x 4
        x = self.features(x)
        x_isup = self.head_isup(x)
        x_gleason_major = self.head_gleason_major(x)
        x_gleason_minor = self.head_gleason_minor(x)
        return(x_isup, x_gleason_major, x_gleason_minor)
