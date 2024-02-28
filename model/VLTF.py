"""
https://github.com/facebookresearch/detr
"""
import copy
from typing import Optional
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import init

class VLTF(nn.Module):
    def __init__(self, v_dim, l_dim=768, num_heads=8, dropout=0.0, num_mem=1, num_neg_mem=1):
        super().__init__()

        self.num_mem = num_mem
        self.num_neg_mem = num_neg_mem
        if num_mem > 0:
            self.memory_token = nn.Embedding(num_mem, v_dim)
        if num_neg_mem > 0:
            self.neg_memory_token = nn.Embedding(num_neg_mem, v_dim)

        self.input_proj = nn.Sequential(
            nn.Conv1d(v_dim, v_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(v_dim),
        )
        self.lan_proj = nn.Sequential(
            nn.Conv1d(l_dim, v_dim, kernel_size=1, stride=1),
        )

        self.vision_lan_fuse = ScaledDotProductAttention(v_dim, h=num_heads, dropout=dropout)
        self.memory_fuse = ScaledDotProductAttention(v_dim, h=num_heads, dropout=dropout)
        self.feature_fuse = ScaledDotProductAttention(v_dim, h=num_heads, dropout=dropout)
        self.norms = nn.ModuleList()
        self.norms.append(nn.InstanceNorm1d(v_dim))
        self.norms.append(nn.InstanceNorm1d(v_dim))
        self.norms.append(nn.InstanceNorm1d(v_dim))

        self.output_proj = nn.Sequential(
            nn.Conv1d(v_dim, v_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(v_dim)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.output_proj[1].weight)
        # nn.init.zeros_(self.neg_memory_token.weight)

    def forward(self, src, lan, pos_embed, lan_attmask):
        B, HW, C = src.shape
        src = self.input_proj(src.permute(0, 2, 1))         # B, C, HW
        lan = self.lan_proj(lan.permute(0, 2, 1))           # B, C, Nl
        
        # pos = pos_embed.flatten(2).permute(0, 2, 1)         # B, HW, C
        lan = lan.permute(0, 2, 1)                          # B, Nl, C
        lan_attmask = lan_attmask.unsqueeze(2)              # B, Nl, 1
        src = src.permute(0, 2, 1)                          # B, HW, C

        vision_lan_fuse, vision_lan_att = self.vision_lan_fuse(lan, src, src, query_mask=lan_attmask)
        vision_lan_fuse = self.norms[0](vision_lan_fuse.permute(0, 2, 1)).permute(0, 2, 1)

        if self.num_mem > 0:
            memory_token = self.memory_token.weight.unsqueeze(0).repeat(B, 1, 1) # B, Nm, C
            # lan_mem = torch.cat([memory_token, lan], dim=1)
            # lan_mem_keymask = torch.cat([torch.ones([B, self.num_mem, 1], device=lan_attmask.device), lan_attmask], dim=1)
            lan_mem, mem_att = self.memory_fuse(memory_token, lan, lan, key_mask=lan_attmask)
            # lan_mem, mem_att = self.memory_fuse(memory_token, vision_lan_fuse, vision_lan_fuse, key_mask=lan_attmask)
        else:
            lan_mem = lan
            mem_att = None
        if self.num_neg_mem > 0:
            neg_memory_token = self.neg_memory_token.weight.unsqueeze(0).repeat(B, 1, 1) # B, Nnm, C
            lan_mem = torch.cat([neg_memory_token, lan_mem], dim=1)
            # lan_attmask = torch.cat([torch.ones([B, self.num_neg_mem, 1], device=lan_mem_keymask.device), lan_mem_keymask], dim=1)
        if lan_mem.shape[1] > 1:
            lan_mem = self.norms[1](lan_mem.permute(0, 2, 1)).permute(0, 2, 1)

        # src, feature_att = self.feature_fuse(src, lan_mem, lan_mem, key_mask=lan_mem_keymask)   # B, HW, C
        src, feature_att = self.feature_fuse(src, lan_mem, lan_mem)   # B, HW, C
        src = self.norms[2](src.permute(0, 2, 1)).permute(0, 2, 1)

        src = self.output_proj(src.permute(0, 2, 1)).permute(0, 2, 1)
            
        return src, lan_mem, vision_lan_att, mem_att, feature_att

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    Modified from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SelfAttention.py
    '''

    def __init__(self, d_model, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_q = nn.Linear(d_model, h * self.d_k)
        self.fc_k = nn.Linear(d_model, h * self.d_k)
        self.fc_v = nn.Linear(d_model, h * self.d_v)
        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, query_mask=None, key_mask=None, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        if query_mask is not None:
            q = q * query_mask.view(b_s, nq, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 1, 3)
        if key_mask is not None:
            k = k * key_mask.view(b_s, nk, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 3, 1)
            v = v * key_mask.view(b_s, nk, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att