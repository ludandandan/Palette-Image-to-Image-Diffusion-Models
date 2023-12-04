from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    checkpoint,
    zero_module,
    normalization,
    count_flops_attn,
    gamma_embedding
)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """

class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input. 是一个顺序模块，可以将EmbedBlock传递给需要它的子模块
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution. # 降采样层，可选卷积
    :param channels: channels in the inputs and outputs. # 输入和输出通道数
    :param use_conv: a bool determining if a convolution is applied. # 是否使用卷积
    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channel, 3, stride=stride, padding=1
            )
        else:# 如果不使用卷积，直接使用平均池化
            assert self.channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of embedding channels.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channels.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channel=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels 
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(# 输入层： 规范化-激活-卷积；channels - self.out_channel
            normalization(channels), # GroupNorm32
            SiLU(),
            nn.Conv2d(channels, self.out_channel, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False) # resblock的输入通道做Downsample模块的输入通道和输出通道，第二个参数False表示不使用卷积，直接使用stride=2的平均池化
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential( # 嵌入层：激活-全连接，emb_channels（256） - self.out_channel
            SiLU(),
            nn.Linear( # 256 ->  self.out_channel
                emb_channels, # 256
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential( # 输出层：规范化-激活-卷积；self.out_channel - self.out_channel
            normalization(self.out_channel), # GroupNorm32
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module( # 将模块的参数归零，梯度与计算图分类（为什么）
                nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
            ),
        )

        if self.out_channel == channels: # 如果输入通道数和输出通道数相同
            self.skip_connection = nn.Identity() # 跳跃连接为恒等映射
        elif use_conv: # 如果输入通道数和输出通道数不同，且use_conv为True
            self.skip_connection = nn.Conv2d( # 跳跃连接为卷积层，将通道数从channels变为self.out_channel
                channels, self.out_channel, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels # 512
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0 # channels必须要被num_head_channels整除
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels # 512 // 32 = 16
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels) # GroupNorm32
        self.qkv = nn.Conv1d(channels, channels * 3, 1) #1维卷积，表示输出数据的通道数。在自注意力机制中，通常需要生成 Q、K 和 V 三个张量，因此输出通道数是输入通道数的三倍，QKV是由一个卷积核shape为（512*3，512，1）的卷积层计算得到的，其中各个参数不同，所以生成的QKV三者不同
        if use_new_attention_order: #如果使用新的注意力顺序
            # split qkv before split heads # 在分割头之前分割qkv
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv # 在分割qkv之前分割头
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1)) # 1维卷积，512->512,将梯度从计算图中分离出来，然后将参数归零

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNet(nn.Module):
    """
    The full UNet model with attention and embedding.
    :param in_channel: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param inner_channel: base channel count for the model.
    :param out_channel: channels in the output Tensor.
    :param res_blocks: number of residual blocks per downsample.
    :param attn_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used. #例如，如果这包含4，则在4倍下采样时将使用注意力。
    :param dropout: the dropout probability.
    :param channel_mults: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channel,
        inner_channel,
        out_channel,
        res_blocks,
        attn_res,
        dropout=0,
        channel_mults=(1, 2, 4, 8),
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    ):

        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads #1

        self.image_size = image_size #256
        self.in_channel = in_channel #6
        self.inner_channel = inner_channel #64
        self.out_channel = out_channel #3
        self.res_blocks = res_blocks #2
        self.attn_res = attn_res#[16]
        self.dropout = dropout #0.2
        self.channel_mults = channel_mults #[1,2,4,8]
        self.conv_resample = conv_resample #True
        self.use_checkpoint = use_checkpoint # False
        self.dtype = torch.float16 if use_fp16 else torch.float32 #torch.float32
        self.num_heads = num_heads #1
        self.num_head_channels = num_head_channels # 32
        self.num_heads_upsample = num_heads_upsample #  1

        cond_embed_dim = inner_channel * 4 #256
        self.cond_embed = nn.Sequential( # 条件嵌入 64 -> 256 -> 256
            nn.Linear(inner_channel, cond_embed_dim), # 64 -> 256 全连接层
            SiLU(), # 激活函数x * torch.sigmoid(x)
            nn.Linear(cond_embed_dim, cond_embed_dim), # 256 -> 256 全连接层
        )

        ch = input_ch = int(channel_mults[0] * inner_channel) # 64, inner_channel = 64 是模型基本通道数，channel_mults = [1,2,4,8] 是UNet的通道倍数
        self.input_blocks = nn.ModuleList( # 输入块：卷积层；ModuleList,用于存储一系列 nn.Module 对象，相比nn.Sequential，它可以存储任意类型的模块，并且没有forward方法，所以不能像nn.Sequential那样直接调用
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))] # 6 -> 64 卷积层，HW不变
        )
        self._feature_size = ch # 64
        input_block_chans = [ch] # [64]
        ds = 1 #下面是进行下采样操作，每次下采样都会将ds乘以2，当ds达到16时，就会使用注意力机制
        for level, mult in enumerate(channel_mults): # channel_mults = [1,2,4,8]，level = 0,1,2,3
            for _ in range(res_blocks): # res_blocks = 2
                layers = [
                    ResBlock(
                        ch, # 64
                        cond_embed_dim, # 256
                        dropout, # 0.2
                        out_channel=int(mult * inner_channel), #乘数*64
                        use_checkpoint=use_checkpoint,# False
                        use_scale_shift_norm=use_scale_shift_norm,# True
                    )
                ]
                ch = int(mult * inner_channel) # 更新ch，用作ResBlock和AttensionBlock的输入通道数
                if ds in attn_res: # attn_res = [16]，看ds是否达到16，当下采样率达到16倍，就使用注意力机制
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers)) # 将定义的ResBlock layer和 AttentionBlock layer添加到输入块列表中
                self._feature_size += ch # 更新_feature_size，用于计算输出块的输入通道数???
                input_block_chans.append(ch) # 将输入块的通道数添加到列表中
            if level != len(channel_mults) - 1: # 如果level不是最后一层，就添加下采样操作
                out_ch = ch
                self.input_blocks.append( #在输入块中加入下ResBlock(down=True)或者下采样层
                    EmbedSequential(
                        ResBlock( #这次的resblock的输入和输出通道一样
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True, # 下采样
                        )
                        if resblock_updown#如果 resblock_updown 为True，就使用ResBlock，否则使用下采样
                        else Downsample(
                            ch, conv_resample, out_channel=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2 # 记录下采样率，每次加倍
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock(
                ch, # 512
                cond_embed_dim, # 256
                dropout, # 0.2
                use_checkpoint=use_checkpoint, # False
                use_scale_shift_norm=use_scale_shift_norm, # True
            ),
            AttentionBlock(
                ch, # 512
                use_checkpoint=use_checkpoint,# False
                num_heads=num_heads, # 1
                num_head_channels=num_head_channels, # 32
                use_new_attention_order=use_new_attention_order,# False
            ),
            ResBlock( # 512 -> 512，与上面的ResBlock一样
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([]) #输出块：上采样操作
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(inner_channel * mult)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channel=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
        )

    def forward(self, x, gammas):
        """
        Apply the model to an input batch.
        :param x: an [N x 2 x ...] Tensor of inputs (B&W)
        :param gammas: a 1-D batch of gammas.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        gammas = gammas.view(-1, )
        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))

        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

if __name__ == '__main__':
    b, c, h, w = 3, 6, 64, 64
    timsteps = 100
    model = UNet(
        image_size=h,
        in_channel=c,
        inner_channel=64,
        out_channel=3,
        res_blocks=2,
        attn_res=[8]
    )
    x = torch.randn((b, c, h, w))
    emb = torch.ones((b, ))
    out = model(x, emb)