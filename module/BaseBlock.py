import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basicconv(x)


class ChannelAttention(nn.Module):

    def __init__(self, channel, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        weight_map = self.sigmoid(x)
        return weight_map


class CA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        x_att = x * channel_attention

        return x_att


class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg_pool, max_pool], dim=1)

        attention_map = self.conv(combined)

        attention_weights = torch.sigmoid(attention_map)

        return attention_weights


class MultiScaleAttentionPyramid(nn.Module):
    def __init__(self, in_channels_x1, in_channels_x2, in_channels_x3):
        super(MultiScaleAttentionPyramid, self).__init__()

        self.attention_x1 = SA()
        self.attention_x2 = SA()
        self.attention_x3 = SA()

        self.conv1x1_x1 = nn.Conv2d(in_channels_x1, in_channels_x2, kernel_size=1)
        self.conv1x1_x12 = nn.Conv2d(in_channels_x2, in_channels_x3, kernel_size=1)

        self.channel_attention = CA((in_channels_x1 + in_channels_x2 + in_channels_x3))

        self.con1x1 = nn.Conv2d((in_channels_x1 + in_channels_x2 + in_channels_x3), in_channels_x3, kernel_size=1)

    def forward(self, x1, x2, x3):
        attention_x1 = self.attention_x1(x1)
        attention_x2 = self.attention_x2(x2)
        attention_x3 = self.attention_x3(x3)

        feature_attention_x1 = attention_x1 * x1
        feature_attention_x2 = attention_x2 * x2
        feature_attention_x3 = attention_x3 * x3

        feature1 = F.interpolate(feature_attention_x1, scale_factor=1 / 2, mode='bilinear', align_corners=True)

        feature12 = torch.cat([feature1, feature_attention_x2], dim=1)
        feature12 = F.interpolate(feature12, scale_factor=1 / 2, mode='bilinear', align_corners=True)
        feature123 = torch.cat([feature12, feature_attention_x3], dim=1)

        feature = self.channel_attention(feature123)
        feature = self.con1x1(feature)

        return feature


class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)

        attention = torch.nn.functional.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out

class DAM(nn.Module):
    def __init__(self, fea_before_channels, fea_rd_channels, out_channels, up=True):
        super(DAM, self).__init__()
        self.up = up

        self.relu = nn.ReLU(inplace=True)
        self.ConvBnRelu = nn.Sequential(
            nn.Conv2d((fea_before_channels + fea_rd_channels * 2), out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.sa = SpatialAttention(kernel_size=7)
        self.ca = ChannelAttention(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        k_size = 3
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(fea_before_channels, out_channels, kernel_size=1)

    def forward(self, fea_before, fea_r, fea_d):
        x = self.ConvBnRelu(torch.cat([fea_r, fea_d, fea_before], dim=1))

        B, C, H, W = x.size()
        P = H * W

        SA = self.sa(x).view(B, -1, P)
        CA = self.ca(x).view(B, C, -1)

        att = torch.bmm(CA, SA).view(B, C, H, W)

        x = x * att

        fea_before_out = self.conv1x1(fea_before)

        out = x + fea_before_out

        if self.up:
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out
