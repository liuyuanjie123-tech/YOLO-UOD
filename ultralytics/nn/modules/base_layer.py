import math
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
import torch
import numpy as np
from ultralytics.nn.modules.ops_dcnv3.models.dcnv3 import DCNv3_pytorch
from ultralytics.nn.modules.wtconv.wtconv2d import WTConv2d
from ultralytics.nn.modules.conv import CBAM, DWConv, ChannelAttention, SpatialAttention
from ultralytics.nn.modules.block import C2f, C3k
from ultralytics.nn.modules.ScConv import ScConv


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad

    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BrokenBlock(nn.Module):
    """ 比起Tensorflow2.x 没想到还是Torch更反人类 """

    def __init__(self, dim_len, dim=1, dim_shape=4, group=1, *args, **kwargs):
        super(BrokenBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.group = group
        self.dim_len = dim_len
        self.view = list(np.ones(dim_shape, int))

    def getShuffle(self, shape):
        # 生成随机索引
        self.view[self.dim] = shape[self.dim]
        perm = torch.randperm(self.dim_len // self.group, device='cuda').unsqueeze(1)
        indices = torch.cat([perm * self.group + _ for _ in range(self.group)], dim=1)
        indices = indices.view(self.view).expand(shape)
        return indices

    def forward(self, x):
        if self.training:
            return torch.gather(x, self.dim, self.getShuffle(x.shape).to(x.device))
        else:
            return x


class PoolBlock(nn.Module):
    def __init__(self, size):
        super(PoolBlock, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(size)
        self.max = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        avg = self.avg(x)
        max = self.max(x)
        return avg + max


class SC(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(SC, self).__init__(*args, **kwargs)
        self.length = length
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = nn.Conv2d(c_ // length, c_ // length, 5, 1, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.brk = BrokenBlock(c_, group=c_ // length)

    def forward(self, x):
        x = self.cv1(x)
        if self.training:
            y = self.brk(x)
        else:
            y = x
        y = y.chunk(self.length, 1)
        out = [self.cv2(_) for _ in y]
        out = self.act(self.bn(torch.cat(out, 1)))
        return torch.cat([x, out], 1)


class ExternalAttention(nn.Module):  # b*1024*20*20
    def __init__(self, c1, c2, k=1, s=1, p=None, g=4, d=1, act=True, down_strid=5, heads=16, dim=1024, size=20):
        super(ExternalAttention, self).__init__()
        heads = math.ceil(heads / (1024 / c1))
        self.down_strid = down_strid
        self.heads = heads
        self.norm1 = nn.GroupNorm(g, heads)
        self.norm2 = nn.BatchNorm2d(heads)
        self.norm3 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU()
        self.act2 = nn.Softmax(dim=-1)
        self.conv_stride1 = nn.Sequential(nn.Conv2d(heads, heads, (1, 5), (1, 5)), self.norm1, self.act1)
        self.conv_stride2 = nn.Sequential(nn.Conv2d(heads, heads, (5, 1), (5, 1)), self.norm1, self.act1)
        self.pwconv1 = nn.Sequential(nn.Conv2d(heads, heads, 3, 1, 1, groups=heads), Conv(heads, 2, 1, 1)
                                     )
        # self.pwconv2 = nn.Sequential(nn.Conv2d(heads, heads, 3, 1, groups=heads),
        #                              nn.Conv2d(heads, heads * 25, 1, 1), self.norm1, self.act1)
        self.linear = nn.Linear(heads, math.ceil(heads * 25 * (1024 / c1)))
        self.conv1 = Conv(dim, c2, 1, 1)
        self.conv2 = Conv(c1 + 2, c1, 1, 1, act=act)
        # 位置坐标
        # self.size = size
        # pos_x, pos_y = torch.meshgrid(torch.arange(1, size + 1), torch.arange(1, size + 1))
        # self.pos_x = nn.Parameter(pos_x.reshape(1, 1, pos_x.shape[-1], pos_x.shape[-1]), requires_grad=False)
        # self.pos_y = nn.Parameter(pos_y.reshape(1, 1, pos_y.shape[-1], pos_y.shape[-1]), requires_grad=False)

    def forward(self, x):  # b*1024*20*20
        # 加上位置编码
        # x = torch.cat([x, self.pos_x, self.pos_y], dim=1)
        # x = self.conv2(x)

        y = rearrange(x, 'b (he c) w h ->b he c w h', he=self.heads)
        c = math.ceil(math.sqrt(y.shape[2]))
        y = rearrange(y, 'b he (c1 c2) w h->b he (w c1) (h c2)', he=self.heads, c1=c, c2=c)  # b*16*160*160
        y1 = self.conv_stride1(y)  # 160*32

        y2 = self.conv_stride2(y)  # 32*160
        y = torch.matmul(y2, y1) / y2.shape[-1]
        y = self.norm2(y)
        # 外部注意力机制
        Mkv = rearrange(self.pwconv1(y), 'b c w h->b c (w h)')
        Mk, Mv = torch.chunk(Mkv, 2, dim=1)
        y = rearrange(y, 'b c w h->b c (w h)')
        w = torch.matmul(y, Mk.transpose(-1, -2)) / y.shape[-1]
        attn = self.act2(w)
        result = torch.matmul(attn, Mv)
        result = result.transpose(-1, -2).contiguous()
        result = self.linear(result)  # b *(16*25)*1024
        result = result.transpose(-1, -2).contiguous()
        result = rearrange(result, 'b (w h) c -> b c w h', w=x.shape[-1], h=x.shape[-1])
        result = self.conv1(result)
        return self.act1(self.norm3(result)) + x if x.shape == result.shape else self.act1(self.norm3(result))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepBlock(nn.Module):
    def __init__(self, c, n, shortcut, g=1, e=0.5):
        super(RepBlock, self).__init__()
        # Bottleneck_DW 是使用了深度可分离卷积，Bottleneck是普通卷积,CSPBottleneck_DW是使用csp思想
        self.m0 = Bottleneck(c, c, shortcut, g=g, e=e)
        self.m = nn.ModuleList(Bottleneck(2 * c, c, shortcut, g=g, e=e) for _ in range(n - 1))

    def forward(self, x):
        device = x.device
        y = []
        a = x.to(device)
        b = self.m0(a)
        y.append(b)
        for m in self.m:
            c = torch.cat([a, b], 1).to(device)
            a = b
            b = m(c)
            y.append(b)

        # return torch.cat(y, dim=1)
        return y


class LPM(nn.Module):  # 这里为了适配YOLO预处理框架，将c1和c2进行互换
    def __init__(self, c1, g=4):  # c1是个list 128 * 160,256 * 80,512 *40
        super(LPM, self).__init__()
        if not isinstance(c1, list):
            c1 = [c1]
        self.scale_attn_module = nn.ModuleList([nn.Sequential(  # scale attention
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(i, 1, (1, 1)),
            nn.GELU()) for i in c1])
        self.dcn1 = DCNv3_pytorch(c1[0], 3, 3, group=g)
        if len(c1) == 3:
            self.dcn2 = DCNv3_pytorch(c1[1], 3, 3, group=g)
            hid = int(c1[1])
            self.DWcov1 = nn.Sequential(
                Conv(c1[0] + c1[1], hid, 1, 1),
                Conv(hid, hid, 3, 1, g=hid)
            )
            self.conv = Conv(c1[-1] + hid, c1[-1], 1, 1)
            self.dcn3 = DCNv3_pytorch(c1[-1], 3, 3, group=g)
            self.DWcov2 = nn.Sequential(
                Conv(c1[-1], c1[-1], 3, 1, g=c1[-1]),
                Conv(c1[-1], c1[-1], 1, 1)
            )
            self.downstrid1 = Conv(c1[0], c1[0], 3, 2, g=c1[0])
            self.downstrid2 = Conv(hid, hid, 3, 2, g=hid)

    def forward(self, x: list):
        x[0] = self.dcn1(x[0])
        x[0] = x[0] * self.scale_attn_module[0](x[0])

        x[1] = self.dcn2(x[1])
        x[1] = x[1] * self.scale_attn_module[1](x[1])

        x[2] = x[2] * self.scale_attn_module[2](x[2])
        # down sampling
        mid_feat1 = self.downstrid1(x[0])
        mid_feat = torch.cat([mid_feat1, x[1]], dim=1)
        mid_feat = self.DWcov1(mid_feat)
        mid_feat = self.downstrid2(mid_feat)
        res = torch.cat([mid_feat, x[-1]], dim=1)
        res = self.conv(res)
        res = self.dcn3(res)
        res = self.DWcov2(res)
        # 与最后一层进行残差连接
        res = res + x[-1]
        return res


class RN(nn.Module):  # input :b*128*160*160 output: b*256*80*80
    def __init__(self, out_channels, scale=32, down_strid=2, act=True, k=3, g=4):  # g=b/2
        super(RN, self).__init__()
        # 使用非深度可分离卷积
        self.scale = scale
        self.down_strid = down_strid
        self.Lconv = nn.Sequential(Conv(out_channels, scale, 3, down_strid, act=act),
                                   Conv(scale, scale, 3, 1, g=scale, act=act))
        self.Mkv = Conv(scale, scale, 3, 1, 1, act=act)  # b*2*20*20
        self.attend = nn.Softmax(dim=-1)
        self.Rconv = Conv(scale, out_channels * down_strid * down_strid, 1, 1, act=act)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = Conv(out_channels, out_channels, 1, 1, act=act)
        self.sc = SC(out_channels, out_channels, k, 1, p=None, g=1, d=1, act=True, length=g)
        self.conv1 = Conv(out_channels * 2, out_channels, 1, 1, act=act)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = Conv(out_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        y = self.Lconv(x)
        height2, width2 = y.shape[-2], y.shape[-1]

        Mkv = self.Mkv(y)
        width = Mkv.shape[-1]
        Mkv = rearrange(Mkv, 'b c h w->b c (h w)')  # b*2*400
        Mk, Mv = Mkv.chunk(2, dim=1)  # b*1*400
        y = rearrange(y, 'b c h w->b c (h w)')  # b*16*400
        # 外部注意力机制
        external_result = torch.matmul(y, Mk.transpose(-1, -2)) / self.scale
        external_result = self.sigmoid(external_result)
        attn = self.attend(external_result)
        external_result = torch.matmul(attn, Mv)
        external_result = rearrange(external_result, 'b c (h w)->b c h w', w=width)  # b*16*20*20
        external_result = self.Rconv(external_result)  # b*256*16*20*20
        external_result = rearrange(external_result, 'b (c s) h w->b c (h w s)',  # b*256*80*80
                                    s=self.down_strid * self.down_strid)
        external_result = rearrange(external_result, 'b c (h w)->b c h w ', w=width2 * self.down_strid)
        #
        if width2 * self.down_strid > x.shape[-1]:  # 若不是整除则需要裁剪为原形状
            external_result = external_result[:, :, 0:x.shape[-2], 0:x.shape[-1]]
        sc_result = self.sc(x)
        return self.conv1(torch.cat([self.conv2(external_result), sc_result], dim=1)) + x


class Coord_attn(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=32, size=1):
        super(Coord_attn, self).__init__()
        self.csp = Conv(in_channels, out_channels)
        self.pool_h, self.pool_w = PoolBlock((size, None)), PoolBlock((None, size))
        c = max(8, in_channels // ratio)
        self.cv1 = nn.Conv2d(in_channels=out_channels, out_channels=c, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(c)
        self.act1 = nn.Hardswish()

        self.cv2 = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.cv3 = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.act2 = nn.Sigmoid()

    def forward(self, x, y=None):
        x = self.csp(x)
        b, c, h, w = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)

        x_tmp = torch.cat([x_h, x_w], dim=-1)
        x_bk = self.act1(self.bn(self.cv1(x_tmp)))

        out_h, out_w = torch.split(x_bk, [h, w], dim=-1)
        out_h = self.cv2(out_h)
        out_w = self.cv3(out_w)
        # out = out_w * out_h.permute(0, 1, 3, 2)
        out = torch.matmul(out_h.permute(0, 1, 3, 2), out_w)

        return x * self.act2(out) if y is None else y * self.act2(out)


class EPM(nn.Module):
    def __init__(self, c1, r=2):  # c1是个list 128 * 160, 256*80,512*40
        super(EPM, self).__init__()
        if not isinstance(c1, list):
            c1 = [c1]
        if len(c1) == 3:
            # 对160做下采样，通道扩张
            self.cv1 = nn.Sequential(Conv(c1[0], c1[0], 3, 2, g=c1[0]),
                                     Conv(c1[0], c1[1], 1, 1))
            # 对40做上采样，通道压缩
            self.cv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                     Conv(c1[2], c1[1], 1, 1))
            # 对80做1*1，通道调整
            self.cv2 = Conv(c1[1], c1[1], 1, 1)
            self.adaptiveAvg = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Sequential(nn.Conv2d(c1[1], c1[1] // r, 1, 1, bias=False),
                                     nn.ReLU(inplace=True))  # 降维
            self.fc2 = nn.Conv2d(c1[1] // r, c1[1] * 3, 1, 1, bias=False)  # 升维
            self.softmax = nn.Softmax(dim=1)

            self.cv4 = Conv(c1[1], c1[1], 1, 1)

    def forward(self, x):
        batch_size = x[0].shape[0]
        x1 = self.cv1(x[0])
        x2 = self.cv2(x[1])
        x3 = self.cv3(x[2])
        output = x1 + x2 + x3
        s = self.adaptiveAvg(output)
        z = self.fc1(s)  # S->Z 降维
        a_b = self.fc2(z)  # Z->a, b 升维 - 论文用 conv 1x1 表示 FC。结果中前一半通道值为 a, 后一半为 b
        a_b = a_b.reshape(batch_size, 3, x1.shape[1], -1)  # reshape 为两个 FCs 的值
        a_b = self.softmax(a_b)  # 令两个 FCs 对应位置进行 softmax
        # the part of selection
        a_b = list(a_b.chunk(3, dim=1))  # split to a 和 b - chunk 将 tensor 按指定维度切分成几块
        out_list = [i.reshape(batch_size, x1.shape[1], 1, 1) for i in a_b]
        x1 = x1 * out_list[0]
        x2 = x2 * out_list[1]
        x3 = x3 * out_list[2]
        out = self.cv4(x1 + x2 + x3)
        return out + x[1]


class WCS_M(nn.Module):

    def __init__(self, c1, c2, n=2, shortcut=False, e=0.5, g=4):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # RGB 多尺度信息
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g=g) for _ in range(n))
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        # 频域多尺度信息 in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1
        self.Wtconv = nn.ModuleList([
            WTConv2d(self.c, self.c, 3, 1, wt_type="coif1"),  # coif1
            WTConv2d(self.c, self.c, 3, 1, wt_type='db1')
        ])
        self.cv3 = Conv(self.c * 2, c2, 1, 1)

        # fuse
        self.fuse = nn.Sequential(Conv(c2 * 3, c2, 1, 1),
                                  Conv(c2, c2, 3, 1, g=c2))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))

        # 频域多尺度
        out_fre = []
        out_fre.extend(self.Wtconv[0](y[0]).unsqueeze(0))
        out_fre.extend(self.Wtconv[1](y[-1]).unsqueeze(0))
        out_fre = self.cv3(torch.cat(out_fre, dim=1))

        # RGB 多尺度
        y.extend(m(y[-1]) for m in self.m)
        out_RGB = self.cv2(torch.cat(y, 1))

        # align
        weight_fuse = torch.mul(out_fre, out_RGB)

        # fuse
        out = self.fuse(torch.cat([out_RGB, out_fre, weight_fuse], dim=1))
        return out


class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.interpolate(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)]
        bottle = torch.cat(priors, 1)
        return self.relu(bottle)


class PBSA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.coord_attn1 = Coord_attn(self.c // 2, self.c // 2, size=1)
        self.dcn = DCNv3_pytorch(self.c, 3, stride=1, pad=autopad(3, 1, 1), group=1, dilation=1, dw_kernel_size=3)
        self.conv1 = nn.Sequential(Conv(self.c, self.c, 3, 1, g=self.c),
                                   Conv(self.c, self.c, 1, 1))
        self.cat = nn.Sequential(Conv(self.c * 3, c1, 1, 1),
                                 Conv(c1, c1, 3, 1, g=c1))
        self.conv2 = Conv(c1, c2, k, s)

    def forward(self, x):
        """Forward pass through C2f layer."""
        # y = list(self.cv1(x).chunk(2, 1))
        y1 = list(self.cv1(x).chunk(2, 1))
        y2 = self.cv2(x)
        # branch1
        out1 = torch.cat([self.coord_attn1(y1[0]), y1[1]], dim=1)
        out1 = self.dcn(out1)
        # branch2
        out2 = self.conv1(y2)
        # aline , 两部分的通道交互
        aline = torch.mul(out1, out2)
        out = self.cat(torch.cat([aline, out1, out2], dim=1))
        # 低通滤波
        out = self.conv2(out + x)
        return out


# ----------------------------------------------------------------------------------------------------
class ChannelShuffleGroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(ChannelShuffleGroupConv, self).__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.shuffle = BrokenBlock(in_channels, group=in_channels // groups)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)  # 第一个分组卷积操作
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.shuffle(x)  # 通道混洗操作
        x = self.conv2(x)  # 第二个分组卷积操作
        x = self.bn2(x)
        x = self.relu2(x)
        return x


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    img1 = torch.randn((1, 128, 160, 160), dtype=torch.float32, device='cuda')
    img2 = torch.randn((1, 512, 80, 80), dtype=torch.float32, device='cuda')
    # img3 = torch.randn((1, 256, 40, 40), dtype=torch.float32, device='cuda')
    # img3 = torch.randn((1, 512, 20, 20), dtype=torch.float32, device='cuda')
    # img4 = torch.randn((1, 1024, 20, 20), dtype=torch.float32, device='cuda:0')
    x = []
    x.append(img1)
    x.append(img2)
    little_attn = EPM([128, 512]).to('cuda')
    res = little_attn(x)
    print(res.shape)
