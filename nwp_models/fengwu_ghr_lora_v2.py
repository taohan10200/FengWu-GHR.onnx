# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
import os
import torch
import numpy as np
from functools import partial
from dict_recursive_update import recursive_update
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from typing import List, Tuple
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from nwp.registry import MODELS
from mmengine.model import BaseModel
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from .lora import LinearLoraGroups, Linear_Head_Groups, Masked_Lora_Linear
from diffusers.models.unets.unet_2d_blocks import get_up_block

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class Conv_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True, mlp_lora=False,lora_rank=16,patch_shape=(60,120)):
        super().__init__()
        self.mlp_lora = mlp_lora
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 3, padding=1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 3, padding=1, bias=bias)
        self.drop = nn.Dropout(drop)
        self.Hp, self.Wp = patch_shape
    def forward(self, 
                x: torch.Tensor, 
                step: torch.Tensor.int = 0,
                ):
        N, D, C = x.shape
        x = x.view(N, self.Hp, self.Wp, -1).permute(0, 3, 1, 2)  
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)      
        x = x.reshape(N, C, self.Hp*self.Wp).permute(0, 2, 1) # NLD     
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True,total_aug_steps=0,mlp_lora=False,lora_rank=16):
        super().__init__()
        self.mlp_lora = mlp_lora
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if mlp_lora:
            self.fc1 = FINETUNE_LINEAR(in_features, hidden_features, lora_rank, total_aug_steps=total_aug_steps)
            self.act = act_layer()
            self.fc2 = FINETUNE_LINEAR(hidden_features, out_features, lora_rank, total_aug_steps=total_aug_steps)
            self.drop = nn.Dropout(drop)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
            self.drop = nn.Dropout(drop)
        
    def forward(self, 
                x: torch.Tensor, 
                step: torch.Tensor.int = 0,
                ):
        if self.mlp_lora:
            x = self.fc1(x, step)
            x = self.act(x)
            # x = self.drop(x)
            x = self.fc2(x,step)
            x = self.drop(x)
        else:
            x = self.fc1(x)
            x = self.act(x)
            # x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)           

        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, window_size=None, rel_pos_spatial=False,total_aug_steps=0,qkv_lora=True,lora_rank=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        if qkv_lora:
            self.qkv = FINETUNE_LINEAR(dim, dim * 3, lora_rank,total_aug_steps=total_aug_steps,bias=qkv_bias) 
        else:
            self.qkv=nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj=nn.Linear(dim, dim)
        self.proj =FINETUNE_LINEAR(dim, dim,lora_rank,total_aug_steps=total_aug_steps) #nn.Linear(dim, dim)     
        self.window_size = window_size

    def forward(self, 
                x: torch.Tensor,
                step: torch.Tensor.int=0
                ):
        B, N, C = x.shape

        if ONNX_EXPORT:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0) # make torchscript happy (cannot use tensor as tuple)   --> (batchsize, heads, len, head_dim)
            attn = ((q * self.scale) @ k.transpose(-2, -1))
            attn = torch.softmax(attn, dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # =====================================
        else:   
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            data_type = qkv.dtype
            qkv=qkv.to(torch.float16) 
            x = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=self.scale, causal=False).reshape(B, N, C)    
            x=x.to(data_type)

        x = self.proj(x, step)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
        attn,
        q,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.

    Source: https://github.com/facebookresearch/mvit/
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, rel_pos_spatial=False, total_aug_steps=0, patch_shape=None,
    qkv_lora=True,lora_rank=16):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial=rel_pos_spatial
        self.Hp, self.Wp = patch_shape

        if qkv_lora:
            self.qkv = FINETUNE_LINEAR(dim, dim*3,lora_rank,total_aug_steps=total_aug_steps, bias=qkv_bias) 
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
        self.proj = FINETUNE_LINEAR(dim, dim,lora_rank, total_aug_steps=total_aug_steps) 
    def forward(self, 
                x: torch.tensor, 
                step: torch.Tensor.int = 0
                ):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
     
        
        x = x.reshape(B_, self.Hp, self.Wp, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - self.Wp % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - self.Hp % self.window_size[0]) % self.window_size[0]
        # assert pad_r == 0 
        # assert pad_b == 0
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)  # num_Windows*B, window_size, window_size, C
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # num_Windows*B, window_size*window_size, C

        B_w = x.shape[0]
        N_w = x.shape[1]
        
        if  ONNX_EXPORT:
            qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  #  --> (batchsize, heads, len, head_dim)
            attn = ((q * self.scale) @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        #=====================================
        else:
            qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads)
            data_type = qkv.dtype
            qkv=qkv.to(torch.float16) 
            x = flash_attn_qkvpacked_func(qkv,dropout_p=0.0, softmax_scale=self.scale, causal=False).reshape(B_w, N_w, C)          
            x=x.to(data_type)
        # ===========================================  
        x = self.proj(x,step)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size, Hp, Wp)  # B H' W' C
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :self.Hp, :self.Wp, :].contiguous()
        
        x = x.view(B_, self.Hp * self.Wp, C)

        return x
    
class Hres_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, window=False, rel_pos_spatial=False,total_aug_steps=0,
                 qkv_lora=False,mlp_lora=False,lora_rank=16,downscale=3, patch_shape=None):
        super().__init__()
        self.norm1_hres = norm_layer(dim)
        self.window = window
        self.downscale = downscale
        self.Hp, self.Wp = patch_shape
        if not window:
            self.attn_hres = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial,
                total_aug_steps = total_aug_steps,
                qkv_lora=qkv_lora,
                lora_rank=lora_rank
                )   
        else:
            self.attn_hres = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial,
                total_aug_steps = total_aug_steps,
                patch_shape = (patch_shape[0]*downscale, patch_shape[1]*downscale),
                qkv_lora=qkv_lora,
                lora_rank=lora_rank
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_hres = norm_layer(dim)
        self.mlp_hres = Conv_Mlp(in_features=dim, 
                                 hidden_features=int(dim),   
                                 act_layer=act_layer,
                                 mlp_lora=mlp_lora,lora_rank=lora_rank,
                                 patch_shape=(patch_shape[0]*downscale, patch_shape[1]*downscale))

    def forward(self, 
                x: torch.Tensor, 
                step: torch.Tensor.int=0,
                ):
       # the embedded self-attention first flattens multi batches of one input to a single two-dimension token and then implements window self-attention. 

        bs_fold, N, C=x.size()  #9, 60*120,3072
        x = x.view(x.size(0), self.Hp, self.Wp,-1).permute(3,0,1,2) ##9, 60, 120, 3072
        
        bs_in =  bs_fold//(self.downscale*self.downscale)
        x = x.contiguous().view(bs_in, C*bs_fold, -1) #9, 3072, 60, 120
        

        x = F.fold(x, 
                   output_size=(self.Hp*self.downscale, self.Wp*self.downscale), 
                   kernel_size=(self.downscale,self.downscale), 
                   stride=(self.downscale,self.downscale)) #  torch.Size([1, 3072,180, 360])   
        
        x = x.flatten(2).transpose(1, 2) #(1,180*360, 3072)
        

        x = x + self.drop_path(self.attn_hres(self.norm1_hres(x), step ))
        x = x + self.drop_path(self.mlp_hres(self.norm2_hres(x), step))
        
        
        x = x.transpose(1, 2).reshape(bs_in, C, self.Hp*self.downscale, self.Wp*self.downscale) #(1,3072, 180, 360)
        x=F.unfold(x, kernel_size=(self.downscale, self.downscale), stride=(self.downscale, self.downscale)) # 1, 3072*9, 60*120
        x = x.view(C, bs_fold, self.Hp, self.Wp) #3072, 9, 60, 120
        
        x = x.permute(1,0,2,3)  #9, 3072, 60, 120
        
        x = x.reshape(x.size(0), x.size(1), -1) # bs*9, 3072, 60*120
        x = x.permute(0, 2, 1).contiguous() 

        #=====batch cross attention finished====================== 
        return (x, step)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, window=False, rel_pos_spatial=False,total_aug_steps=0,
                 qkv_lora=True,mlp_lora=False,lora_rank=16,downscale=3, norm_finetune=False, patch_shape=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.window = window
        self.downscale = downscale
        self.norm_finetune = norm_finetune
        self.Hp, self.Wp = patch_shape
        
        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial,
                total_aug_steps = total_aug_steps,
                qkv_lora=qkv_lora,
                lora_rank=lora_rank
                )   
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial,
                total_aug_steps = total_aug_steps,
                patch_shape = patch_shape,
                qkv_lora=qkv_lora,
                lora_rank=lora_rank
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, total_aug_steps=total_aug_steps , mlp_lora=mlp_lora, lora_rank=lora_rank)
   
    def duplicted_norm_weights(self):
        if hasattr(self, 'norm1_groups'):
            for i in range(len(self.norm1_groups)):
                self.norm1_groups[i].load_state_dict(self.norm1.state_dict())     
        if hasattr(self, 'norm2_groups'):
            for i in range(len(self.norm2_groups)):
                self.norm2_groups[i].load_state_dict(self.norm2.state_dict()) 

    def forward(self, 
                x: torch.Tensor, 
                step: torch.Tensor.int=0, 
                ):

        x = x + self.drop_path(self.attn(self.norm1(x),step ))
        x = x + self.drop_path(self.mlp(self.norm2(x), step)) 
        # print(x)             
        return (x, step)

class PatchEmbed_SIM(nn.Module):
    
    """ A Patch embedidng layer that embeds the Spatial Indentical Mapping (SIM) strategy.
    """
    def __init__(self, img_size=224, patch_size=16,patch_stride=16, in_chans=3, embed_dim=768,
                 total_aug_steps=0, lora_rank=0, pos_embed=torch.nn.Parameter, downscale=int):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.downscale = downscale
        self.patch_shape = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])  # could be dynamic
        
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride)
        self.pos_embed = pos_embed

    def duplicted_step_customized_param(self):
        if hasattr(self, 'proj_groups'):
            for i in range(len(self.proj_groups)):
                self.proj_groups[i].load_state_dict(self.proj.state_dict())    
                      
    def forward(self, 
                x: torch.Tensor, 
                step: torch.Tensor.int=0,
                **kwargs): 

        # if  x.requires_grad:
        #     x=x.detach()     
        # bs_in, c_in, h_in, w_in = x.size()
  
        
        # x=F.unfold(x, kernel_size=(self.downscale, self.downscale), stride=(self.downscale, self.downscale)) # 1, 69*9, 721*1440
        # x = x.view(c_in, -1, self.img_size[0], self.img_size[1]) # (74, 9, 721, 1440)
        # x = x.permute(1,0,2,3)  # (9, 74, 721, 1440)
    
        if self.training:     
            if not x.requires_grad:
                x.requires_grad=True
        
        x = self.proj(x) #step
        # import pdb
        # pdb.set_trace()
        bs_in, c_in, h_in, w_in = x.size()
        x=F.unfold(x, kernel_size=(self.downscale, self.downscale), stride=(self.downscale, self.downscale)) # 1, 69*9, 721*1440
        x = x.view(c_in, -1, h_in//self.downscale, w_in//self.downscale) # (74, 9, 721, 1440)
        x = x.permute(1,0,2,3)  # (9, 74, 721, 1440)

        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed 
        return (x, step)


            
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class Decoder(nn.Module):
    def __init__(self, down_linear, final, up_blocks, norm, ending_norm, patch_shape, img_size, downscale):
        super().__init__()
        self.norm = norm
        self.down_linear  = down_linear
        self.final = final
        self.up_blocks = up_blocks
        self.ending_norm = ending_norm
        self.Hp,self.Wp = patch_shape
        self.img_size = img_size
        self.downscale = downscale
        # self.filter = nn.Conv2d(74, 74, kernel_size=3, stride=1, padding=1)
    def forward(self, 
                x:torch.Tensor, 
                step:torch.Tensor.int=0,
                ):
        
        if self.ending_norm:
            x = self.norm(x)  # b h*w c
        x = self.down_linear(x)   
        x = x.view(x.size(0), self.Hp, self.Wp,-1)
        x = x.permute(0, 3, 1, 2)

        bs, c, h_, w_ = x.size() #(9, 3072, 60, 120)
        x = x.permute(1,0,2,3).contiguous().view(1, bs*c, -1) #(1, 9x3700, 60x120)
        x=F.fold(x, 
                 output_size=(h_*self.downscale, w_*self.downscale), 
                 kernel_size=(self.downscale,self.downscale), 
                 stride=(self.downscale,self.downscale)) #  torch.Size([1, 69,721*3,1440*3])    

        for blk in self.up_blocks:
            x =blk(x)


        if  self.img_size==(721, 1440):
            x = self.final(x)
            # x = checkpoint( self.filter, x)          
        else:
            x = x.permute(0,2,3,1)
            x = self.final(x)
            x = rearrange(
                x,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=3, # self.patch_size[-2],
                p2=3, # self.patch_size[-1],
                h=self.img_size[0] // 3, # self.patch_size[-2],
                w=self.img_size[1] // 3, # self.patch_size[-1],
            )  
        
        return (x, step)
    
class FengWu_GHR(BaseModel):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 patch_stride=16, 
                 in_chans=3, 
                 out_chans=227, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 window_size=(14,14),
                 drop_path_rate=0., 
                 norm_layer=None, 
                 window=True,
                 use_abs_pos_emb=False, 
                 interval=3,
                 test_pos_mode='simple_interpolate',
                 learnable_pos=False,
                 rel_pos_spatial=False,
                 lms_checkpoint_train=False,
                 lms_checkpoint_layer_interval=1,
                 pad_attn_mask=False, 
                 freeze_iters=0,
                 act_layer='GELU', 
                 pre_ln=False, 
                 mask_input=False, 
                 ending_norm=True,
                 round_padding=False, 
                 compat=False,
                 total_aug_steps=0,
                 lora_rank=16, 
                 qkv_lora=True,
                 mlp_lora=False, 
                 downscale=3, 
                 across_window_size=(60,120),
                 lora_name = 'LinearLoraGroups',
                 onnx_export = False
                 ):
        super().__init__()

        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.lms_checkpoint_layer_interval = lms_checkpoint_layer_interval
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding
        self.patch_size = patch_size
        self.img_size = img_size
        self.total_aug_steps = total_aug_steps
        self.depth = depth
        self.num_heads =num_heads
        self.Hp, self.Wp = img_size[0] // patch_stride[0], img_size[1] // patch_stride[1]
        self.ori_Hp, self.ori_Hw = img_size[0] // patch_size[0], \
                                   img_size[1] // patch_size[1]
        self.qkv_lora=qkv_lora
        self.mlp_lora=mlp_lora
        self.lora_rank=lora_rank
        self.downscale = downscale
        self.block_interval = interval
        global COMPAT, ONNX_EXPORT
        COMPAT = compat
        ONNX_EXPORT = onnx_export
        global FINETUNE_LINEAR        
        if lora_name == 'LinearLoraGroups':
            FINETUNE_LINEAR = LinearLoraGroups
        elif lora_name == 'Linear_Head_Groups':
            FINETUNE_LINEAR = Linear_Head_Groups
        elif lora_name == 'Masked_Lora_Linear':
            FINETUNE_LINEAR = Masked_Lora_Linear
        else:
            raise ValueError(f"{lora_name} must be 'LinearLoraGroups or Linear_Head_Groups")
        
            
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.Wp*self.Hp, embed_dim), requires_grad=learnable_pos)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.Wp, self.Hp), cls_token=False)

            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            raise
        
        self.patch_embed = PatchEmbed_SIM(
            img_size=img_size, patch_size=patch_size, patch_stride= patch_stride,
            in_chans=in_chans, embed_dim=embed_dim, 
            total_aug_steps=self.total_aug_steps, lora_rank = self.lora_rank, 
            pos_embed = self.pos_embed, downscale = self.downscale)

        num_patches = self.patch_embed.num_patches
        


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        for i in range(depth):
            which_win = min(i%interval, len(window_size)-1)
            block = Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i], 
                norm_layer=norm_layer,                
                window_size=window_size[which_win] if ((i + 1) % interval != 0) else self.patch_embed.patch_shape,
                window=((i + 1) % interval != 0) if window else False,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU,
                total_aug_steps =  total_aug_steps,
                qkv_lora = self.qkv_lora,
                mlp_lora = self.mlp_lora,
                lora_rank=lora_rank,
                downscale = self.downscale,
                patch_shape = (self.Hp, self.Wp),
            )
            self.blocks.append(block)
            
        assert window_size is not None  
        self.blocks_hres = nn.ModuleList()
        for i in range(depth//6): #interval):
            block =Hres_Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop_path=dpr[i], norm_layer=norm_layer,                
            window_size= across_window_size , 
            window=True,
            rel_pos_spatial=rel_pos_spatial,
            act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU,
            total_aug_steps =  total_aug_steps,
            qkv_lora = self.qkv_lora,
            mlp_lora = self.mlp_lora,
            lora_rank = lora_rank,
            downscale = self.downscale,
            patch_shape = (self.Hp, self.Wp),
            )
            self.blocks_hres.append(block)

        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only
        self.norm = norm_layer(embed_dim)

        # up
        self.down_linear = nn.Linear(self.embed_dim, 1024, bias=False)

        # if self.patch_size[-1] == 10:
        #     block_out_channels= [384, 512]  #24 ,96 ,384
        # else:
        #     block_out_channels= [256, 384, 512]  #24 ,96 ,384
        if self.patch_size[-1] == 10:
            block_out_channels= [768, 1024]  #24 ,96 ,384
        else:
            block_out_channels= [256, 768, 1024]  #24 ,96 ,384
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel =  output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                "UpDecoderBlock2D",
                num_layers=1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                attention_head_dim=output_channel,
                temb_channels=None,
                resnet_time_scale_shift="group",
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        up_count = len(block_out_channels) - 1
        ps_left = self.patch_size[0]//(2**up_count)
        
        if self.img_size==(721, 1440):
            self.final = nn.ConvTranspose2d(in_channels=block_out_channels[0], 
                                out_channels=out_chans,
                                kernel_size=(ps_left+1, ps_left),                    
                                stride=(ps_left, ps_left), 
                                bias=False)

        else:
            self.final = nn.Linear(block_out_channels[0], out_chans*ps_left*ps_left, bias=False)
        self.decoder=Decoder(
                            self.down_linear,
                            self.final, 
                            self.up_blocks,
                            self.norm, 
                            self.ending_norm,
                            (self.Hp, self.Wp), 
                            self.img_size, 
                            self.downscale,
                             )
        
        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()


    def duplicted_step_customized_param(self):
        if hasattr(self, 'final_groups'):
            for i in range(len(self.final_groups)):
                self.final_groups[i].load_state_dict(self.final.state_dict(),strict=False) 
                if self.final_groups[i].bias is not None:
                    nn.init.constant_(self.final_groups[i].bias, 0)
        if hasattr(self, 'patch_embed'):
            self.patch_embed.duplicted_step_customized_param() 
        
        if hasattr(self, 'norm_groups'):
            for i in range(len(self.norm_groups)):
                self.norm_groups[i].load_state_dict(self.norm.state_dict())            
        
        for block in self.blocks:
            block.duplicted_norm_weights()
            
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        if hasattr(self, 'input_hres'): 
            nn.init.zeros_(self.input_hres.weight)
        if hasattr(self, 'final_hres'): 
            nn.init.zeros_(self.final_hres.weight)
        
        for layer_id, layer in enumerate(self.blocks):
            if isinstance(layer, Block):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
 
        for layer_id, layer in enumerate(self.blocks_hres):
            rescale(layer.attn_hres.proj.weight.data, layer_id + 1)
            rescale(layer.mlp_hres.fc2.weight.data, layer_id + 1)
            if hasattr(layer, 'mlp_hres'):
                nn.init.constant_(layer.mlp_hres.fc2.weight.data, 0)
                nn.init.constant_(layer.mlp_hres.fc2.bias.data, 0)
            if hasattr(layer, 'attn_hres'): 
                nn.init.constant_(layer.attn_hres.proj.weight, 0)    
                nn.init.constant_(layer.attn_hres.proj.bias, 0)  
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _normalization(x):
        assert len(x.shape) == 4
        x = x.sub(torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1).cuda()).div(torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1).cuda())
        return x

    def get_num_layers(self):
        return len(self.blocks)


    def transformer_forward(self, 
                            x: torch.Tensor, 
                            step: torch.Tensor.int=0, 
                            ):
        x = (x, step)
        for i, blk in enumerate(self.blocks):
            if self.lms_checkpoint_train:
                if  self.training:
                    x = checkpoint(blk, *x)
                else:
                    x = blk(*x)
            else:
                x = blk(*x)
            
            if  (i+1) % 6 == 0 and self.downscale!=1: #self.block_interval == 0: # the wrong implementation is '(i+1) // self.block_interval==0' before 05/03/2024  
                if self.lms_checkpoint_train:
                    x = checkpoint(self.blocks_hres[i//6], *x)
                else:
                    x = self.blocks_hres[i//6](*x)
        return x

    def forward(self, 
                x: torch.Tensor, 
                step: torch.Tensor.int=0, 
                **kwargs):
        _, _, h_in, w_in = x.shape
        
        if h_in != self.img_size[0]*self.downscale or w_in!=self.img_size[1]*self.downscale:
            x = F.interpolate(x.float(), 
                              size=(self.img_size[0]*self.downscale, self.img_size[1]*self.downscale), 
                              mode='bilinear')

         #1, 69, 2160,4320

        ###====================================
        x = self.patch_embed( x, step, **kwargs)
        x = self.transformer_forward(*x)
        x = self.decoder(*x)[0]
        #  ================================
        
        if h_in != self.img_size[0]*self.downscale or w_in!=self.img_size[1]*self.downscale:
            x = F.interpolate(x.float(), 
                              size=(h_in, w_in), 
                              mode='bilinear')     
        
        return x
    
    def init_weights(self, pretrained='',):
        from mmengine.runner.checkpoint import load_from_ceph      
        from collections import OrderedDict     
        import os
        if len(pretrained)>0:
            Unexpected  = []
            if pretrained.endswith('.tar'):
                pretrained_dict = torch.load(pretrained)['state_dict']
                print('=> loading pretrained model {}'.format(pretrained))
                model_dict = self.state_dict()
                pretrained_dict_filter ={}
                for k, v in pretrained_dict.items():
                    if k[23:] in model_dict.keys() and "pos_embed" not in k:
                        pretrained_dict_filter.update({k[23:]: v})
                        
            elif pretrained.endswith('.pth'):
                if 's3:' in pretrained:
                    print(f'{self.__class__.__name__} is loading checkpoint from ceph:{pretrained}')
                    checkpoint = load_from_ceph(pretrained, map_location='cpu')
                else:
                    print(f'{self.__class__.__name__} is loading checkpoint from local:{pretrained}')
                    checkpoint = torch.load(pretrained, map_location='cpu')
                if 'model' in checkpoint:
                    pretrained_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    pretrained_dict = checkpoint['state_dict']
                else:
                    raise ValueError
                
                print('=> loading pretrained model {}'.format(pretrained))
                model_dict = self.state_dict()

                pretrained_dict_filter = OrderedDict()

                for k, v in pretrained_dict.items():
                    k = k.replace('backbone.arch_net.', '')
                    if k in model_dict.keys(): # and "pos_embed" not in k and "patch_embed" not in k:
                        pretrained_dict_filter.update({k: v})
                    else:
                        Unexpected.append(k)
                Missing = list(set(model_dict) - set(pretrained_dict_filter))
                Missing.sort()
            print(
                "Missing keys: {}".format(Missing),
                "Unexpected keys: {}".format(Unexpected),
                )

            self.load_state_dict(pretrained_dict_filter)



class dummy_logger:
    def info(self, **kwargs):
        print(**kwargs)

    def warning(self, **kwargs):
        print(**kwargs)



def load_checkpoint(model, state_dict, load_pos_embed, strict=False, logger=None):
    """
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    # if not isinstance(checkpoint, dict):
    #     raise RuntimeError(
    #         f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'pos_embed' in state_dict:
        if load_pos_embed:
            state_dict['pos_embed'] = interpolate_pos_embed(pos_embed_checkpoint=state_dict['pos_embed'],
                                                            patch_shape=model.patch_embed.patch_shape,
                                                            num_extra_tokens=1)
        else:
            del state_dict['pos_embed']
            print("checkpoint pos_embed removed")

    model_dict = model.state_dict()
    load_dict = {
        k: v for k, v in state_dict.items() if k in model_dict.keys()
    }
    print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))

    load_state_dict(model, state_dict, strict, logger)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')


    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    print("finish load")



# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    # import pdb
    # pdb.set_trace()
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_abs_pos(abs_pos, has_cls_token, ori_hw, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    embed_num, _, emde_dim = abs_pos.size()
    h, w = hw
    if has_cls_token:
     abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))

    ori_hp, ori_hw = ori_hw

    assert ori_hp, ori_hw == xy_num

    if ori_hp != h or ori_hw != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(embed_num, ori_hp, ori_hw, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1).reshape(embed_num, h*w, -1)
    else:
        return abs_pos.reshape(embed_num, h*w, -1)

class FengWu_Hres_Lora_v2(BaseModel):
    def __init__(self, 
                 arch='vit_base', 
                 patch_size=(16,16),
                 patch_stride=None, 
                 in_chans=227,
                 out_chans=227,
                 pretrained_model=None, 
                 finetune_model=None,
                 kwargs=None):
        super().__init__()

        if patch_stride is None:
            patch_stride =patch_size

        base_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=patch_size, patch_stride=patch_stride, in_chans=in_chans, out_chans=out_chans, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True,  

            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # learnable_pos= True,
        )

        large_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=patch_size,patch_stride=patch_stride,in_chans=in_chans, out_chans=out_chans, embed_dim=1024,         depth=24,
            num_heads=16, mlp_ratio=4, qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # learnable_pos= True,
        )


        huge_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=patch_size,patch_stride=patch_stride,in_chans=in_chans, out_chans=out_chans, embed_dim=3072, depth=24,
            num_heads=16, mlp_ratio=4, qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # learnable_pos= True,
        )

        B10_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=patch_size,patch_stride=patch_stride,in_chans=in_chans, out_chans=out_chans, embed_dim=8192, depth=36,
            num_heads=64, mlp_ratio=4, qkv_bias=True,  
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # learnable_pos= True,
        )
        
        if arch == "vit_base":
            recursive_update(base_default_dict, kwargs)
            self.arch_net = FengWu_GHR(**base_default_dict)

        elif arch == "vit_large":

            recursive_update(large_default_dict, kwargs)
            self.arch_net = FengWu_GHR(**large_default_dict)

        elif arch == "vit_huge":
            recursive_update(huge_default_dict, kwargs)
            self.arch_net = FengWu_GHR(**huge_default_dict)
        elif arch == "vit_10B":
            recursive_update(B10_default_dict, kwargs)
            self.arch_net = FengWu_GHR(**B10_default_dict)
        else:
            raise Exception("Architecture undefined!")


        if  pretrained_model is not None:
            self.arch_net.init_weights(pretrained_model)

        
        if finetune_model is not None:
            import io
            pretrained_dict = torch.load(finetune_model, map_location='cpu')['state_dict']
            # with open(pretrained_model, 'rb') as f:
            #     buffer = io.BytesIO(f.read())
            # checkpoint = torch.load(buffer)
            # checkpoint = clip_checkpoint_preprocess(checkpoint)

            # load while interpolates position embedding
            # import pdb
            # pdb.set_trace()
            model_dict = self.state_dict()
            pretrained_dict_filter ={}
            for k, v in pretrained_dict.items():
                if k[9:] in model_dict.keys():
                    pretrained_dict_filter.update({k[9:]: v})
            load_state_dict(self.arch_net, pretrained_dict_filter, strict=False, logger=dummy_logger)
            print(
                "Missing keys: {}".format(list(set(model_dict) - set(pretrained_dict_filter)
                                               )))
            model_dict.update(pretrained_dict_filter)
            # import pdb
            # pdb.set_trace()
            self.load_state_dict(model_dict)
            del pretrained_dict

    def forward(self, 
                input:torch.Tensor, 
                step:torch.Tensor.int=0,
                **kwargs):
        return  self.arch_net(input, step, **kwargs)

