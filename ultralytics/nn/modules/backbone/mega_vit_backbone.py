from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from ultralytics.nn.modules.backbone.Blocks import Block_Mega
from ultralytics.nn.modules.backbone.util import get_2d_sincos_pos_embed
import torch.nn.functional as F
from thop import profile


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class Backbone_MegaVit(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, out_indices=None,
                 num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, use_moe=True, num_expert=4):
        super().__init__()

        self.img_size = img_size
        self.use_moe = use_moe
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        if self.use_moe:
            self.blocks = nn.ModuleList([
                Block_Mega(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=True, norm_layer=norm_layer,
                           use_moe=use_moe, num_expert=num_expert)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)])

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        b, _, c = x.shape

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        feature = []
        loss_moe = 0.
        for i, blk in enumerate(self.blocks):
            if self.use_moe:
                x, loss_l = blk(x)
                loss_moe += loss_l
            else:
                x = blk(x)

            if i in self.out_indices:
                feature.append(x[:, 1:, ].permute(0, 2, 1).view(b,
                                                                c,
                                                                self.img_size // self.patch_size,
                                                                self.img_size // self.patch_size).contiguous())

        return feature, loss_moe

    def forward(self, imgs):
        x, loss_moe = self.forward_encoder(imgs)
        return x, loss_moe


def backbone_base(img_size=640):
    model = Backbone_MegaVit(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        # out_indices=[3, 5, 7, 11],
        out_indices=[5, 7, 11],
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_moe=True,
        num_expert=4,
    )
    return model


def backbone_large(img_size):
    model = Backbone_MegaVit(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        # out_indices=[7, 11, 15, 23],
        out_indices=[11, 15, 23],
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_moe=True,
        num_expert=4,
    )
    return model


class VitBackbone(nn.Module):
    def __init__(self, img_size=640, pretrain_path=None):
        super(VitBackbone, self).__init__()
        self.encoder = backbone_base(img_size=img_size)

        self.pretrain_path = pretrain_path
        if self.pretrain_path is not None:
            self.load_pretrain()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=self.encoder.embed_dim, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.encoder.embed_dim),
            nn.ReLU(inplace=True))

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.encoder.embed_dim, out_channels=self.encoder.embed_dim, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.encoder.embed_dim),
            nn.ReLU(inplace=True))

    def load_pretrain(self):
        checkpoint = torch.load(self.pretrain_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % self.pretrain_path)
        checkpoint_model = checkpoint['model']
        interpolate_pos_embed(self.encoder, checkpoint_model)
        self.encoder.load_state_dict(checkpoint_model, strict=False)

    def forward(self, x):
        feature, loss_moe = self.encoder(x)
        x1 = self.up(feature[0])
        x2 = feature[1]
        x3 = self.down(feature[2])

        return [[x1, x2, x3], loss_moe]
        # return [x1, x2, x3]


class split_scale(nn.Module):
    def __init__(self, flag=0):
        super(split_scale, self).__init__()
        self.flag = flag

    def forward(self, scaleList: list):
        # 优化：直接断言长度等于3
        assert len(scaleList[0]) == 3, "VitBackBone的输出不是三个尺度"
        return scaleList[0][int(self.flag)]


if __name__ == '__main__':
    torch.cuda.set_device(7)
    x = torch.randn(1, 3, 640, 640).cuda()
    model = VitBackbone(img_size=640,
                        pretrain_path=None).cuda()
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    split_l = split_scale(flag=1)
    res = split_l(out)
    print(res.shape)
    flops, params = profile(model, inputs=(x,))
    print("parms=M", params / (1000 ** 2))
    print("flops=G", flops / (1000 ** 3))
