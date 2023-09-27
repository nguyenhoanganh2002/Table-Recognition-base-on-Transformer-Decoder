
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch
from torchinfo import summary
from dataclasses import dataclass
from torch.nn import functional as F
from PIL import Image
from torchvision import models


class ResNet_with_FPN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config.d
        self.resnet = resnet18()
        self.pyramid = []
        self.return_nodes = {
            'layer1.1.conv2': 'layer1', #P2
            'layer2.1.conv2': 'layer2', #P3
            'layer3.1.conv2': 'layer3', #P4
            'layer4.1.conv2': 'layer4'  #P5
        }
        self.feature_extractor = create_feature_extractor(self.resnet, return_nodes=self.return_nodes)
        self.conv1_list = nn.ModuleList([
            nn.Conv2d(64, 1, (1,1), 1),
            nn.Conv2d(128, 1, (1,1), 1),
            nn.Conv2d(256, 1, (1,1), 1),
            nn.Conv2d(512, 1, (1,1), 1),
        ])
        self.conv3_list = nn.ModuleList([nn.Conv2d(1, self.d, (3,3), 2) for _ in range(4)])
        
    def forward(self, x):
        B, C, H, W = x.shape
        intermediate_outputs = self.feature_extractor(x)
        pyramid = []
        outs = []
        for i in range(3, -1, -1):
            layer_name = 'layer'+str(i)
            conv1 = self.conv1_list[i]
            conv3 = self.conv3_list[i]
            pyramid[i] = conv1(intermediate_outputs[layer_name])
            if i<3:
                pyramid[i] += F.interpolate(pyramid[i+1], scale_factor=(2,2))
            outs[i] = conv3(pyramid[i])
        return outs

class MultiAspectGCAttention(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio=0.065,
                 headers=1,
                 pooling_type='att',
                 att_scale=False,
                 fusion_type='channel_concat'):
        super().__init__()
        assert pooling_type in ['avg', 'att']

        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        assert inplanes % headers == 0 and inplanes >= 8  # inplanes must be divided by headers evenly

        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = False

        self.single_header_inplanes = int(inplanes / headers)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
            # for concat
            self.cat_conv = nn.Conv2d(2 * self.inplanes, self.inplanes, kernel_size=1)
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            # [N*headers, C', H , W] C = headers * C'
            x = x.view(batch * self.headers, self.single_header_inplanes, height, width)
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            # input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.view(batch * self.headers, self.single_header_inplanes, height * width)

            # [N*headers, 1, C', H * W]
            input_x = input_x.unsqueeze(1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.view(batch * self.headers, 1, height * width)

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / torch.sqrt(torch.tensor(self.single_header_inplanes))

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x

        if self.fusion_type == 'channel_mul':
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape

            out = torch.cat([out, channel_concat_term.expand(-1, -1, H, W)], dim=1)
            out = self.cat_conv(out)
            out = nn.functional.layer_norm(out, [self.inplanes, H, W])
            out = nn.functional.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_gcb=False, gcb_config=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        self.use_gcb = use_gcb

        if self.use_gcb:
            gcb_ratio = gcb_config['ratio']
            gcb_headers = gcb_config['headers']
            att_scale = gcb_config['att_scale']
            fusion_type = gcb_config['fusion_type']
            self.context_block = MultiAspectGCAttention(inplanes=planes,
                                                        ratio=gcb_ratio,
                                                        headers=gcb_headers,
                                                        att_scale=att_scale,
                                                        fusion_type=fusion_type)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_gcb:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, gcb=None, in_channels=1):
        super(ResNet, self).__init__()
        gcb_config = gcb

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][0])

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][1])

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][2])

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.layer4 = self._make_layer(block, 256, layers[3], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][3])

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_gcb=False, gcb_config=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_gcb=use_gcb, gcb_config=gcb_config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.layer3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.layer4(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        return x


def resnet50(gcb_kwargs, in_channels=1):
    model = ResNet(BasicBlock, [1, 2, 5, 3], gcb=gcb_kwargs, in_channels=in_channels)
    return model


class ConvEmbeddingGC(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        gcb_kwargs = {
                    "ratio": 0.0625,
                    "headers": 8,
                    "att_scale": True,
                    "fusion_type": "channel_concat",
                    "layers":[False, True, True, True]
                }
        self.backbone = resnet50(gcb_kwargs, in_channels=in_channels)

    def forward(self, x):
        feature = self.backbone(x)
        b, c, h, w = feature.shape  # （B， C， H/8, W/4）
        feature = feature.view(b, c, h * w)
        feature = feature.permute((0, 2, 1))
        return feature

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.pe_dropout)

        position = torch.arange(config.pe_maxlen).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.pe = torch.zeros(config.pe_maxlen, 1, config.n_embd)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)].transpose(0,1)
        return self.dropout(x)

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.c_attn_q = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)   # q * Wq
        self.c_attn_kv = nn.Linear(self.n_embd, self.n_embd*2, bias=config.bias)# k,v * Wk,
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        # self.mask = torch.tril(torch.ones((self.block_size, self.block_size))).view(1, 1, self.block_size, self.block_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)


    def forward(self, x, encoder_output):
        if x.shape[0] != encoder_output.shape[0]:
            print("error ca")
            return
        assert x.shape[0] == encoder_output.shape[0]
        B, T, C = encoder_output.shape                                  # batch_size, block_size, n_embd
        B, N, C = x.shape                                               # batch_size, n_queries, n_embd

        k, v = self.c_attn_kv(encoder_output).split(self.n_embd, dim=2)
        q = self.c_attn_q(x)

        q = q.view(B, N, self.n_heads, C//self.n_heads).transpose(1,2)  # (B, n_heads, N, h_size)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)  # (B, n_heads, T, h_size)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)  # (B, n_heads, T, h_size)

        att = q@k.transpose(-2,-1)*(1/math.sqrt(k.size(-1)))            # (B, n_heads, N, T)
        att = F.softmax(att, dim=-1)                                    # (B, n_heads, N, T)
        att = self.attn_dropout(att)
        y = att@v                                                       # (B, n_heads, N, h_size)
        y = y.transpose(1,2).contiguous().view(B,N,C)                   #(B, N, n_heads*h_size)
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class SelfAttention(nn.Module):
    def __init__(self, config, masked=False, inp="tag"):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        if inp == "tag":
            self.block_size = config.tags_maxlen
        else:
            self.block_size = config.content_maxlen
        self.c_attn = nn.Linear(self.n_embd, self.n_embd*3, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.mask = torch.tril(torch.ones((self.block_size, self.block_size))).view(1, 1, self.block_size, self.block_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.masked = masked


    def forward(self, x):
        B, T, C = x.shape                                               #batch_size, block_size, n_embd
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)  # (B, n_heads, T, h_size)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)

        att = q@k.transpose(-2,-1)*(1/math.sqrt(k.size(-1)))            # (B, n_heads, T, T)
        if self.masked:
            att = att.masked_fill_(self.mask[:, :, :T, :T] == 0, -float('inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att@v                                                       #(B, n_heads, T, h_size)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, config, masked=True, inp="tag"):
        super().__init__()
        self.self_attention = SelfAttention(config, masked, inp)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.cross_attention = CrossAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
    def forward(self, prev_output, encoder_output):
        x = self.self_attention(prev_output)
        x = self.ln_1(x)
        x = x + self.cross_attention(x, encoder_output)
        x = self.ln_2(x)
        x = x + self.mlp(x)
        x = self.ln_3(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
        self.pe = PositionalEncoding(config)

    def forward(self, x):
        x = self.backbone(x)            #(B, n_emb, H, H)
        x = x.flatten(start_dim=2)      #(B, n_emb, H*H)
        x = x.transpose(1,2)            #(B, H*H, n_emb)
        outs = self.pe(x)               #(B, H*H, n_emb)
        return outs
    
class SharedDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.tags_vocab_size, config.n_embd)
        self.pe = PositionalEncoding(config)
        self.decoders = nn.ModuleList([DecoderBlock(config, masked=True, inp="tag")] + \
                        [DecoderBlock(config, masked=False, inp="tag")]*(config.n_decoder_blocks - 1))

    def forward(self, x, encoder_output):
        x = x.to(torch.int64)
        x = self.emb(x)                     # B, tags_maxlen, n_emb
        x = self.pe(x)
        for layer in self.decoders:
            x = layer(x, encoder_output)
        return x
    
class StructureDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoders = nn.ModuleList([DecoderBlock(config, masked=False)]*config.n_decoder_blocks)
        self.fc_out = nn.Linear(config.n_embd, config.tags_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output):
        for layer in self.decoders:
            x = layer(x, encoder_output)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x
    
class BBoxDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoders = nn.ModuleList([DecoderBlock(config, masked=False)]*config.n_decoder_blocks)
        self.fc_out = nn.Linear(config.n_embd, 4)
        self.dropout = nn.Dropout(config.dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, encoder_output):
        for layer in self.decoders:
            x = layer(x, encoder_output)
        x = self.fc_out(x)
        x = self.dropout(x)
        outs = self.sigmoid(x)
        return outs

class ContentDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.emb = nn.Embedding(config.content_vocab_size, self.n_embd)
        self.pe = PositionalEncoding(config)
        self.decoders = nn.ModuleList([DecoderBlock(config, masked=True, inp="content")] + \
                        [DecoderBlock(config, masked=False)]*(config.n_decoder_blocks - 1))
        self.fc_out = nn.Linear(config.n_embd, config.content_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, decoder_output):
        B, tags_maxlen, content_maxlen = x.shape    # B, tags_maxlen, content_maxlen
        x = x.reshape(B*tags_maxlen, content_maxlen)   # B*tags_maxlen, content_maxlen
        x = x.to(torch.int64)
        x = self.emb(x).float()                         # B*tags_maxlen, content_maxlen, n_emb
        x = self.pe(x)
        x += decoder_output.unsqueeze(2).repeat(1,1,content_maxlen,1).view(B*tags_maxlen, content_maxlen, self.n_embd)
        encoder_output = encoder_output.unsqueeze(1).repeat(1,tags_maxlen,1,1).view(B*tags_maxlen, encoder_output.shape[1], self.n_embd)

        for layer in self.decoders:
            x = layer(x, encoder_output)

        x = self.fc_out(x)
        x = self.dropout(x)

        return x.view(B, content_maxlen, tags_maxlen, x.shape[-1]).transpose(1,2)
