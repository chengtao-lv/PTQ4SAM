import math
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from ptq4sam.quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401
from projects.instance_segment_anything.models.segment_anything.modeling.transformer import Attention, TwoWayAttentionBlock
from projects.instance_segment_anything.models.segment_anything.modeling.common import MLPBlock
from ptq4sam.quantization.quantized_module import PreQuantizedLayer,QuantizedMatMul

from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import Attention as EncoderAttention
from projects.instance_segment_anything.models.segment_anything.modeling.transformer import Attention as DecoderAttention
from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import Block, ImageEncoderViT
from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import window_partition, window_unpartition, add_decomposed_rel_pos


def update_specialized_quantizer_config(base_config, quantizer_name):
    import copy
    specialized_config = copy.deepcopy(base_config)

    update_keys = {
        'softmax':{'quantizer':'AdaptiveGranularityQuantize',
                   'observer':'LogAvgMSEFastObserver'},
        'bimodal':{'quantizer':'LSQSignFakeQuantize',
                   'observer':'SignAvgMSEFastObserver'}
    }[quantizer_name]
    specialized_config.update(update_keys)
    return specialized_config


class QuantEncoderAttentionBlock(QuantizedBlock):
    def __init__(self, org_module: EncoderAttention, w_qconfig, a_qconfig, qoutput=False, qinput=True):
        super().__init__()
        self.qinput = qinput
        self.qoutput = qoutput
        self.num_heads = org_module.num_heads
        self.scale = org_module.scale

        self.qkv = PreQuantizedLayer(org_module.qkv, None, w_qconfig, a_qconfig)
        self.proj = PreQuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig)
        self.use_rel_pos = org_module.use_rel_pos

        if self.use_rel_pos:
            self.rel_pos_h = org_module.rel_pos_h
            self.rel_pos_w = org_module.rel_pos_w
        
        self.softmax_post_act_fake_quantize = Quantizer(None, a_qconfig)

        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)
        
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        
        # print(q.shape, k.shape, v.shape, self.num_heads);exit()
        q = self.q_post_act_fake_quantize(q)
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = self.softmax_post_act_fake_quantize(attn.softmax(dim=-1))
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class QuantNeck(QuantizedBlock):
    def __init__(self, org_module: nn.Sequential, w_qconfig, a_qconfig, qoutput=True, qinput=False):
        super().__init__()
        org_module[0] = PreQuantizedLayer(org_module[0], None, w_qconfig, a_qconfig)
        org_module[2] = PreQuantizedLayer(org_module[2], None, w_qconfig, a_qconfig)
        self.model = org_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class QuantMLPBlock(QuantizedBlock):
    def __init__(self, org_module: MLPBlock, w_qconfig, a_qconfig, qinput=True):
        super().__init__()
        self.lin1 = PreQuantizedLayer(org_module.lin1, None, w_qconfig, a_qconfig)
        self.lin2 = PreQuantizedLayer(org_module.lin2, None, w_qconfig, a_qconfig)
        self.act = org_module.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class QunatEncoderBlock(nn.Module):
    def __init__(self, org_module: Block, w_qconfig, a_qconfig, qoutput=True ) -> None:
        super().__init__()
        self.norm1 = org_module.norm1
        self.attn = QuantEncoderAttentionBlock(org_module.attn, w_qconfig, a_qconfig)
        self.norm2 = org_module.norm2
        self.mlp = QuantMLPBlock(org_module.mlp, w_qconfig, a_qconfig)

        self.window_size = org_module.window_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class QuantImageEncoderViT(nn.Module):
    
    def __init__(self, org_module: ImageEncoderViT, w_qconfig, a_qconfig, qoutput=True ) -> None:
        super().__init__()
        self.img_size = org_module.img_size
        self.patch_embed = org_module.patch_embed
        # do not quantize the first block/layer pos_embed
        self.pos_embed = org_module.pos_embed
        
        self.blocks = nn.ModuleList()
        for i in range(len(org_module.blocks)):
            self.blocks.append(QunatEncoderBlock(org_module.blocks[i], w_qconfig, a_qconfig))

        self.neck = QuantNeck(org_module.neck, w_qconfig, a_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class QuantDecoderOurTwoWayAttentionBlock(nn.Module):
    
    def __init__(self, org_module: TwoWayAttentionBlock, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True ) -> None:
        super().__init__()
        self.self_attn = QuantDecoderOurAttentionBlock(org_module.self_attn, w_qconfig, a_qconfig, ptq4sam_config, qinput=True)
        self.norm1 = org_module.norm1

        self.cross_attn_token_to_image = QuantDecoderOurAttentionBlock(
            org_module.cross_attn_token_to_image, w_qconfig, a_qconfig, ptq4sam_config, qinput=True
        )
        self.norm2 = org_module.norm2

        self.mlp = QuantMLPBlock(org_module.mlp, w_qconfig, a_qconfig, qinput=True)
        self.norm3 = org_module.norm3

        self.norm4 = org_module.norm4
        self.cross_attn_image_to_token = QuantDecoderOurAttentionBlock(
            org_module.cross_attn_image_to_token, w_qconfig, a_qconfig, ptq4sam_config, qinput=True
        )

        self.skip_first_layer_pe = org_module.skip_first_layer_pe
    
    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn((queries, queries, queries))
        else:
            q = queries + query_pe
            attn_out = self.self_attn((q, q, queries))
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        # attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        attn_out = self.cross_attn_token_to_image((q, k, keys))        
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token((k, q, queries))
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class QuantDecoderOurAttentionBlock(QuantizedBlock):
    def __init__(self, org_module: DecoderAttention, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True, qinput=False):
        super().__init__()
        self.qoutput = qoutput
        self.embedding_dim = org_module.embedding_dim
        self.internal_dim = org_module.internal_dim
        self.num_heads = org_module.num_heads
        
        self.q_proj = PreQuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig)
        self.k_proj = PreQuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig)
        self.v_proj = PreQuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig)
        self.out_proj = PreQuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig)
        # self.out_proj = QuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig, qoutput=False)

        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, 'softmax')
        else:
            softmax_a_config = a_qconfig
        if ptq4sam_config.BIG:
            sign_a_config = update_specialized_quantizer_config(a_qconfig, 'bimodal')
        else:
            sign_a_config = a_qconfig
        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)
        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, sign_a_config)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

        if ptq4sam_config.BIG:
            self.k_post_act_fake_quantize.global_num = ptq4sam_config.global_num
            self.k_post_act_fake_quantize.peak_distance = ptq4sam_config.peak_distance
            self.k_post_act_fake_quantize.peak_height = ptq4sam_config.peak_height

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    # def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    def forward(self, qkv: tuple) -> Tensor:

        q,k,v = qkv[0],qkv[1],qkv[2]

        # Input projections
        q = self.q_post_act_fake_quantize(self.q_proj(q))
        k = self.k_post_act_fake_quantize(self.k_proj(k))
        v = self.v_post_act_fake_quantize(self.v_proj(v))

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        
        attn = self.softmax_post_act_fake_quantize(attn,value=v)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    
    def bimodal_adjust(self):
        if self.k_post_act_fake_quantize.is_bimodal:
            sign = self.k_post_act_fake_quantize.sign
            def addjust_linear(linear:torch.nn.Linear, sign):
                linear.weight.mul_(sign.unsqueeze(1))
                linear.bias.mul_(sign)
            addjust_linear(self.k_proj.module, sign)
            addjust_linear(self.q_proj.module, sign)
            self.k_post_act_fake_quantize.is_bimodal = False


class QuantImageEncoderOurViT(nn.Module):
    
    def __init__(self, org_module: ImageEncoderViT, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True ) -> None:
        super().__init__()
        self.img_size = org_module.img_size
        self.patch_embed = org_module.patch_embed
        # do not quantize the first block/layer pos_embed
        self.pos_embed = org_module.pos_embed
        
        self.blocks = nn.ModuleList()
        for i in range(len(org_module.blocks)):
            self.blocks.append(QunatEncoderOurBlock(org_module.blocks[i], w_qconfig, a_qconfig, ptq4sam_config))

        self.neck = QuantNeck(org_module.neck, w_qconfig, a_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

class QunatEncoderOurBlock(nn.Module):
    def __init__(self, org_module: Block, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True ) -> None:
        super().__init__()
        self.norm1 = org_module.norm1
        self.attn = QuantEncoderOurAttentionBlock(org_module.attn, w_qconfig, a_qconfig, ptq4sam_config)
        # print(self.attn)
        self.norm2 = org_module.norm2
        self.mlp = QuantMLPBlock(org_module.mlp, w_qconfig, a_qconfig)

        self.window_size = org_module.window_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class QuantEncoderOurAttentionBlock(QuantizedBlock):
    def __init__(self, org_module: EncoderAttention, w_qconfig, a_qconfig, ptq4sam_config, qoutput=False, qinput=True):
        super().__init__()
        self.qinput = qinput
        self.qoutput = qoutput
        self.num_heads = org_module.num_heads
        self.scale = org_module.scale

        self.qkv = PreQuantizedLayer(org_module.qkv, None, w_qconfig, a_qconfig)
        self.proj = PreQuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig)
        self.use_rel_pos = org_module.use_rel_pos

        if self.use_rel_pos:
            self.rel_pos_h = org_module.rel_pos_h
            self.rel_pos_w = org_module.rel_pos_w
        
        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig,'softmax')
        else:
            softmax_a_config = a_qconfig
        
        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)

        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        q = self.q_post_act_fake_quantize(q)
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = self.softmax_post_act_fake_quantize(attn.softmax(dim=-1), value=v)
        
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

specials = {
    TwoWayAttentionBlock: QuantDecoderOurTwoWayAttentionBlock,
    Attention: QuantDecoderOurAttentionBlock,
    ImageEncoderViT: QuantImageEncoderOurViT,
}

def bimodal_adjust(model,logger):
    logger.info('start to detect dimodal distribution')
    for name,m in model.named_modules():
        if isinstance(m, QuantDecoderOurAttentionBlock) and 'token_to_image' in name:
            logger.info(name)
            # print(m.k_post_act_fake_quantize.is_A_two_peak, m.k_post_act_fake_quantize.is_bimodal)
            logger.info(m.k_post_act_fake_quantize.is_bimodal)
            m.bimodal_adjust()
    logger.info('bimodal integration end')