import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()
        self.sensors = para.num_sensors
        _para = deepcopy(para)
        _para.num_channels = int(_para.num_channels / 2)
        self.stream0 = MSTCN(para)
        self.stream1 = MSTCN(_para)
        self.stream2 = MSTCN(_para)
        classes = para.num_classes
        _para.num_channels = 3 * classes
        _para.num_stages = 1
        self.fusion_stages = MSTCN(_para)
        self.ca = ChannelAttention(_para.num_channels)
        self.only_last_layer = False

    def forward(self, x):
        x = x.permute(0, 2, 1)
        acceleration = [6 * i + j for i in range(self.sensors) for j in range(3)]
        rotation = [6 * i + j for i in range(self.sensors) for j in range(3, 6)]
        x0 = x
        x1 = x[:, acceleration, :].clone()
        x2 = x[:, rotation, :].clone()
        outputs = []
        out0 = self.stream0(x0)
        out1 = self.stream1(x1)
        out2 = self.stream2(x2)
        outputs.append(out0)
        outputs.append(out1)
        outputs.append(out2)
        out = [F.softmax(out0[:, :, :, -1].squeeze(dim=-1), dim=1),
               F.softmax(out1[:, :, :, -1].squeeze(dim=-1), dim=1),
               F.softmax(out2[:, :, :, -1].squeeze(dim=-1), dim=1)]
        self.streams = [s.detach().squeeze().cpu().numpy() for s in out]
        out = torch.cat(out, dim=1)
        out = self.ca(out)
        out = self.fusion_stages(out)
        outputs.append(out)
        if self.only_last_layer:
            return torch.cat(outputs, dim=-1)[:, :, :, -1].squeeze(dim=-1)
        else:
            return torch.cat(outputs, dim=-1)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpppl = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.cw_linear = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Linear(2 * channels, channels),
            nn.ReLU()
        )

    def forward(self, x):
        cw_avg = self.cw_linear(self.avgpool(x).squeeze(dim=-1))
        cw_max = self.cw_linear(self.avgpool(x).squeeze(dim=-1))
        cw = self.sigmoid(cw_avg + cw_max).unsqueeze(dim=-1)
        self.cw = cw.detach().squeeze().cpu().numpy()
        out = cw * x

        return out


class MSTCN(nn.Module):
    def __init__(self, para):
        super(MSTCN, self).__init__()
        stages = para.num_stages
        channels = para.num_channels
        classes = para.num_classes
        feats = para.num_feats
        layers = para.num_layers
        self.stages = nn.ModuleList()
        self.stages.append(SingleStage(channels, classes, feats, layers))
        for i in range(stages - 1):
            self.stages.append(SingleStage(classes, classes, feats, layers))

    def forward(self, x):
        outputs = []
        out = x
        for stage in self.stages:
            out = stage(out)
            outputs.append(out.unsqueeze(dim=-1))
            out = F.softmax(out, dim=1)

        return torch.cat(outputs, dim=-1)


class SingleStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_feats, num_layers):
        super(SingleStage, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, num_feats, kernel_size=1)
        self.layers = nn.ModuleList([DilatedResBlock(2 ** i, num_feats, num_feats) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_feats, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)

        return out


class DilatedResBlock(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResBlock, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.30)

    def forward(self, x):
        out = self.relu(self.conv_dilated(x))
        out = self.dropout(self.conv1x1(out))
        out = out + x

        return out
