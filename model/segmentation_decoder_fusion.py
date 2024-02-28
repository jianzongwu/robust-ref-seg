from multiprocessing import pool
from re import L
import numpy as np
import torch
from torch.functional import Tensor 
import torch.nn as nn 
import torch.nn.functional as F
from typing import List
import math

from VLTF import ScaledDotProductAttention
from msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder

class SegmentationDecoderFusion(nn.Module):
    def __init__(self, config, args, position_encoding):
        super().__init__()
        self.args = args
        model_name = config.MODEL.NAME
        if model_name == 'swin':
            embed_dim = config.MODEL.SWIN.EMBED_DIM
            vis_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
            num_heads = config.MODEL.SWIN.NUM_HEADS
        elif model_name == 'segformer':
            vis_channels = [64, 128, 320, 512]
            num_heads = [1, 2, 5, 8]
        elif model_name == 'convnext':
            vis_channels = [128, 256, 512, 1024]
            num_heads = [1, 2, 4, 8]
        else:
            vis_channels = [256,512,1024,2048]
            num_heads = [4, 8, 16, 32]

        self.cross_attns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projects = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        for i in range(len(vis_channels)):
            if i > 0:
                self.cross_attns.append(ScaledDotProductAttention(vis_channels[i], h=num_heads[i], dropout=0.0))
                norm = nn.BatchNorm1d(vis_channels[i])
                nn.init.zeros_(norm.weight)
                self.norms.append(norm)
            self.mask_convs.append(nn.Conv2d(vis_channels[i], 2, 1))
            if i == len(vis_channels) - 1:
                continue
            else:
                channel = vis_channels[i + 1] + vis_channels[i]
                self.projects.append(MultiScaleProj(channel, vis_channels[i]))   
        
        if args.use_pixel_decoder:
            self.pixel_decoder = MSDeformAttnPixelDecoder(in_channels=vis_channels)
            self.mask_conv = nn.Conv2d(256, 2, 1)
            self.exist_pred_channel = 256
            self.num_heads = 8
        # self.exist_pred_channel_index = 3 - self.args.n_fuse + 1
        self.exist_pred_channel_index = 1
        self.exist_pred_channel = vis_channels[self.exist_pred_channel_index]
        self.num_heads = num_heads[self.exist_pred_channel_index]
        if args.use_exist:
            self.src_cross_attn = ScaledDotProductAttention(self.exist_pred_channel, h=self.num_heads, dropout=0.0)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.exist_pred = nn.Linear(self.exist_pred_channel, 1)

        # position_encoding
        self.position_encoding = position_encoding

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pixel_decoder(self, feature_list):
        for i in range(len(feature_list)):
            B, HW, C = feature_list[i].shape
            H = W = int(math.sqrt(HW))
            feature_list[i] = feature_list[i].view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        mask_feature, multi_scale_features = self.pixel_decoder(feature_list)
        return mask_feature
    
    def forward(self, feature_list, memory_list, lan_attmask):
        if self.args.use_pixel_decoder:
            # pass
            mask_feature = self.forward_pixel_decoder(feature_list)
            mask_list = []
            mask_list.insert(0, self.mask_conv(mask_feature))
            B, C, H, W = mask_feature.shape
            identity = mask_feature.permute(0, 2, 3, 1).view(B, H * W, C)
        else:
            out = feature_list[-1]      # B, HW, C
            mask_list = []
            for i in reversed(range(len(feature_list))):
                # if i > (3 - self.args.n_fuse):
                if i > 0 and i <= self.args.n_fuse:
                    identity = out
                    out, att = self.cross_attns[i - 1](out, memory_list[i], memory_list[i])
                    out = identity + self.norms[i - 1](out.permute(0, 2, 1)).permute(0, 2, 1)   # B, HW, C

                B, HW, C = out.shape
                H = W = int(math.sqrt(HW))
                out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W

                mask = self.mask_convs[i](out)                  # B, 2, H, W
                mask_list.insert(0, mask)

                if i > 0:
                    out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
                    next_feature = feature_list[i - 1]
                    B, HW, C = next_feature.shape
                    H = W = int(math.sqrt(HW))
                    next_feature = next_feature.permute(0, 2, 1).view(B, C, H, W)
                    out = torch.cat([out, next_feature], dim=1)
                    out = self.projects[i - 1](out)
                    out = out.permute(0, 2, 3, 1).view(B, H * W, C)

        if self.args.use_exist:
            mem = memory_list[self.exist_pred_channel_index].detach()
            identity = identity.detach()
            # exist_feature = out_list[0].transpose(1, 2).detach()
            exist_feature, att = self.src_cross_attn(identity, mem, mem)
            exist_feature = exist_feature.transpose(1, 2)
            pool_feature = self.avgpool(exist_feature)
            pool_feature = pool_feature.flatten(1)
            exist_pred = torch.sigmoid(self.exist_pred(pool_feature))
        else:
            exist_pred = None
            att = None

        return {
            "mask_list": mask_list,
            "exist_pred": exist_pred,
            "att": att
        }

class MultiScaleProj(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(C_out)
        self.norm2 = nn.BatchNorm2d(C_out)
    def forward(self, x):
        return F.relu(self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))))

# class ExistClassification(nn.Module):
#     def __init__(self, C, hidden_dim, num_mem):
#         super().__init__()
#         self.num_mem = num_mem
#         self.linear1 = nn.Linear(C, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, 2)
#     def forward(self, memory_list, lan_attmask):
#         all_memory = torch.cat(memory_list, dim=2)              # B, Nl, 4C
#         B, Nl, C = all_memory.shape
#         lan_attmask = lan_attmask.unsqueeze(2)
#         lan_mem_mask = torch.cat([torch.ones([B, self.num_mem, 1], device=lan_attmask.device), lan_attmask], dim=1)
#         all_memory = all_memory * lan_mem_mask
#         logit = self.linear2(F.relu(self.linear1(all_memory)))  # B, Nl, 2
#         result = torch.mean(logit, 1)
#         return result