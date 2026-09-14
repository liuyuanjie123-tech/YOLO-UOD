import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:  # 做一次或多次空洞卷积 ，计算padding 的大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class C2fAMSP(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, 1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)  # m每执行一次y.extend加一个
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


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


class SpiralConv(nn.Module):  # 仅进行Amsp没有VC
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g=4, act=act)
        self.brk = BrokenBlock(c_, group=c_ // 4)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        y = self.brk(x)
        return self.cv3(torch.cat((x, self.cv2(y)), 1))


class DW(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, *args, **kwargs):
        super(DW, self).__init__(*args, **kwargs)
        self.cv1 = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1)
        self.cv2 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.cv1(x))
        y = self.act(self.bn(self.cv2(y)))
        return y


class SCL(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(SCL, self).__init__(*args, **kwargs)
        self.length = length
        self.c = in_channels // length
        self.cv1 = nn.Conv2d(self.c, self.c, 3, 1, 1)
        self.cv2 = nn.Conv2d(self.c, self.c, k, s, autopad(k, p, d), d)
        self.cv3 = nn.Conv2d(in_channels, out_channels, bias=False)

        self.brk = BrokenBlock(in_channels, group=self.c)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        if self.training:
            y = x.chunk(self.length, 1)
            y = [self.act(self.cv1(_)) for _ in y]
            x = self.brk(torch.cat(y, 1))
            z = x.chunk(self.length, 1)
            z = [self.act(self.cv2(_)) for _ in z]
            z = torch.cat(z, 1)
        else:
            x = x.chunk(self.length, 1)
            y = [self.act(self.cv1(_)) for _ in x]
            z = [self.act(self.cv2(_)) for _ in y]
            z = torch.cat(z, 1)
        out = self.act(self.bn(self.cv3(z)))
        return out


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


class SCADDNoise(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, length=4, bn_ratio=0.4, ln_ratio=0.6,
                 max_size=640, *args,
                 **kwargs):
        super(SCADDNoise, self).__init__(*args, **kwargs)
        self.length = length
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = nn.Conv2d(c_ // length, c_ // length, 5, 1, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.brk = BrokenBlock(c_, group=c_ // length)
        # # # 新加的
        # noise = torch.randn((1, self.noise_channel, max_size, max_size)) / 255
        # self.maxsize = max_size
        # self.noise = nn.Parameter(torch.clamp(noise, 0, 1 / 255), requires_grad=False)
        self.cv3 = Conv(c_, c_, 1, 1, act=act)
        # self.bn_ratio = bn_ratio
        # self.ln_ratio = ln_ratio

        # self.bn_ratio = nn.Parameter(torch.rand(1, dtype=torch.half), requires_grad=True)
        # self.ln_ratio = nn.Parameter(torch.rand(1, dtype=torch.half), requires_grad=True)
        # self.cv3 = Conv(c_,c_,1,1,None,g,act=act)

    def forward(self, x):
        x = self.cv1(x)
        if self.training:
            y = self.brk(x)
        else:
            y = x

        # 加高斯噪声  1/4图片的1/4个通道加噪声
        noise_channel = math.ceil(x[1] / 8)
        shape_n = math.ceil(y.shape[0] / 8)
        noise_channel = self.noise_channel
        noise = torch.randn((shape_n, self.noise_channel, x.shape[-1], x.shape[-1]), device=x.device) / 255
        noise_empty_channel = torch.zeros((shape_n, y.shape[1] - noise_channel, y.shape[-1], y.shape[-1]),
                                          device=x.device)
        noise = torch.cat([noise, noise_empty_channel], dim=1)
        empty_noise = torch.zeros((y.shape[0] - shape_n, y.shape[1], y.shape[-1], y.shape[-1]),
                                  device=x.device)
        noise = torch.cat([noise, empty_noise], dim=0)

        if y.shape == noise.shape and self.training:
            y = y + noise
            y = self.cv3(y)

        y = y.chunk(self.length, 1)
        out = [self.cv2(_) for _ in y]  # shared weights
        temp = torch.cat(out, 1)
        # layerNorm = torch.nn.LayerNorm(temp.shape[1:], elementwise_affine=False, device=x.device)
        # out = self.act(self.bn(temp)*self.bn_ratio + layerNorm(temp)*self.ln_ratio)
        out = self.act(self.bn(temp))
        # 更新 经过一层 1*1卷积
        # out = self.cv3(y)
        return torch.cat([x, out], 1)


class LinearConv(nn.Module):  # 1,64,320,320
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, length=2, *args, **kwargs):
        super(LinearConv, self).__init__(*args, **kwargs)
        self.length = length
        # 2次下采样，3次上采样
        c_ = int(c1 / length)  # 32
        o1, o2, o3, o4 = int(c1 / 2), int(c1 / 4), int(c1 / 2), c1  # output channel  32,16,32,64,128
        self.cv1 = Conv(c_, o1, k, s, None, g, act=act)
        self.cv1_1 = Conv(o1 * 2, o1, 1, 1)
        self.cv2 = nn.Conv2d(int(o1 / length), o2, 1, 1, bias=False)
        self.cv2_2 = Conv(o2 * 2, o2, (1, 1), (1, 1))  # 16
        self.cv3 = nn.Conv2d(int(o2 / length), o3, 1, 1, bias=False)
        self.cv3_3 = Conv(o3 * 2, o3, 1, 1)  # 32
        self.cv4 = Conv(o3, o4, 1, 1)
        self.cv5 = Conv(o4, c2, 1, 1)

    def forward(self, x):
        s = self.length
        y = x.chunk(s, 1)
        out = [self.cv1(_) for _ in y]
        out = self.cv1_1(torch.cat(out, 1))

        y = out.chunk(s, 1)
        out = [self.cv2(_) for _ in y]
        out = self.cv2_2(torch.cat(out, 1))

        y = out.chunk(s, 1)
        out = [self.cv3(_) for _ in y]
        out = self.cv3_3(torch.cat(out, 1))
        y = self.cv5(self.cv4(out) + x)
        return y


class BKSC(nn.Module):  # 直接对应维度相加
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(BKSC, self).__init__(*args, **kwargs)
        self.length = length
        self.cv1 = Conv(c1, c2, k, s, None, length, act=act)
        self.cv2 = nn.Conv2d(c2 // length, c2 // length, 5, 1, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.brk = BrokenBlock(c2, group=c2 // length)

    def forward(self, x):
        x = self.cv1(x)
        if self.training:
            y = self.brk(x)
        else:
            y = x
        y = y.chunk(self.length, 1)
        out = [self.cv2(_) for _ in y]
        out = self.act(self.bn(torch.cat(out, 1)))
        return out + x


class Spiral(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(Spiral, self).__init__(*args, **kwargs)
        self.length = length
        self.act = nn.SiLU()

        self.cv1 = nn.Conv2d(in_channels, in_channels, k, s, autopad(k, p, d), d, groups=length, bias=False)
        self.brk = BrokenBlock(dim_len=in_channels, group=in_channels // length)
        self.cv2 = nn.Conv2d(in_channels, in_channels, 5, 1, 2, 1, groups=length, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.cv3 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.act(self.cv1(x))
        if self.training:
            y = self.brk(y)
        y = self.cv2(y)
        y = self.bn2(y)
        y = self.act(y)
        if x.shape == y.shape:
            out = self.act(self.bn3(self.cv3(y + x)))
        else:
            out = self.act(self.bn3(self.cv3(y)))
        return out


class PoolBlock(nn.Module):
    def __init__(self, size):
        super(PoolBlock, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(size)
        self.max = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        avg = self.avg(x)
        max = self.max(x)
        return avg + max


class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=16):  # 32
        super(PWConv, self).__init__()
        self.csp = Conv(in_channels, out_channels)
        self.pool_h, self.pool_w = PoolBlock((1, None)), PoolBlock((None, 1))
        c = max(8, in_channels // ratio)
        self.cv1 = nn.Conv2d(in_channels=out_channels, out_channels=c, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(c)
        self.act1 = nn.Hardswish()

        self.cv2 = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.cv3 = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.csp(x)
        b, c, h, w = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)

        x_tmp = torch.cat([x_h, x_w], dim=-1)
        x_bk = self.act1(self.bn(self.cv1(x_tmp)))

        out_h, out_w = torch.split(x_bk, [h, w], dim=-1)
        out_h = self.cv2(out_h)
        out_w = self.cv3(out_w)
        out = out_w * out_h.permute(0, 1, 3, 2)

        return x * self.act2(out)


if __name__ == '__main__':
    # matrix = torch.randn()
    # method = PWConv()
    # print(matrix)
    # test = BrokenBlock(matrix.shape, 2)
    # matrix = test(matrix)
    # print(matrix)

    # 输入张量
    input = torch.tensor([[10, 20],
                          [30, 40],
                          [50, 60]])

    # 索引张量
    index = torch.tensor([[0, 1],
                          [1, 0],
                          [1, 1]])

    # 在第一个维度上进行 gather 操作
    output = torch.gather(input, dim=-1, index=index)

    print(output)
