''' Depth-guided Deep Video Inpainting
'''
from unittest import result
import numpy as np
import time
import math
from operator import mul
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single
import torchvision
import torchvision.models as models
from core.spectral_norm import spectral_norm as _spectral_norm


# #############################################################################
# Depth-guided Deep Video Inpainting
# #############################################################################

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

# stage-1: depth completion
class DepthCompletion(BaseNetwork):
    def __init__(self, init_weights=True):
        super(DepthCompletion, self).__init__()
        channel_d = 64
        hidden_d = 256
        fusion_dim_d = 32
        stack_num = 8
        num_head = 4
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        blocks = []
        dropout = 0.
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)
        for _ in range(stack_num):
            blocks.append(TransformerBlock(hidden=hidden_d, dim_hidden=fusion_dim_d, num_head=num_head, dropout=dropout, n_vecs=n_vecs, t2t_params=t2t_params))
        self.transformer = nn.Sequential(*blocks)
        self.ss_depth = SoftSplit(channel_d, hidden_d, kernel_size, stride, padding, dropout=dropout)
        self.add_pos_emb_depth = AddPosEmb(n_vecs, hidden_d)
        self.sc_depth = SoftComp(channel_d, hidden_d, output_size, kernel_size, stride, padding)
        self.encoder_depth = DepthEncoder(channel_d)

        # decoder: decode frames from features
        self.decoder_depth = nn.Sequential(
            deconv(channel_d, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        if init_weights:
            self.init_weights()

    def forward(self, depth_frames, masks):
        # extracting features
        b, t, c, h, w = depth_frames.size()
        time0 = time.time()    
        depth_frames = depth_frames.view(b * t, 1, h, w)
        masks = (1. - masks).view(b * t, 1, h, w)
        enc_feat_depth = self.encoder_depth(torch.cat((depth_frames,masks), dim=1))
        # trans_feat for depth channel
        trans_feat_depth = self.ss_depth(enc_feat_depth, b)
        trans_feat_depth = self.add_pos_emb_depth(trans_feat_depth)
        _,_,c2 = trans_feat_depth.size()
        # transformer body
        trans_feat_depth = self.transformer(trans_feat_depth)
        trans_feat_depth = self.sc_depth(trans_feat_depth, t)
        trans_feat_depth = enc_feat_depth + trans_feat_depth
        pred_depths = self.decoder_depth(trans_feat_depth)
        return pred_depths

# stage-2: content reconstruction    
class ContentReconstruction(BaseNetwork):
    def __init__(self, init_weights=True):
        super(ContentReconstruction, self).__init__()
        channel_c = 128
        hidden_c = 512
        channel_d = 32
        hidden_d = 128
        fusion_dim_c = 40
        fusion_dim_d = 16
        stack_num = [4,4,4]
        num_head = 4
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        blocks1 = []
        blocks2 = []
        dropout = 0.
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)
        for _ in range(stack_num[0]):
            blocks1.append(TransformerBlockWithDepth(hidden1=hidden_c, hidden2=hidden_d, dim_hidden1=fusion_dim_c, dim_hidden2=fusion_dim_d, num_head=num_head, dropout=dropout, n_vecs=n_vecs, t2t_params=t2t_params))
        self.transformer1 = nn.Sequential(*blocks1)
        for _ in range(stack_num[1]):
            blocks2.append(TransformerBlock(hidden=hidden_c, dim_hidden=fusion_dim_c, num_head=num_head, dropout=dropout, n_vecs=n_vecs, t2t_params=t2t_params))
        self.transformer2 = nn.Sequential(*blocks2)
        self.ss = SoftSplit(channel_c, hidden_c, kernel_size, stride, padding, dropout=dropout)
        self.ss_depth = SoftSplit(channel_d, hidden_d, kernel_size, stride, padding, dropout=dropout)
        self.add_pos_emb = AddPosEmb(n_vecs, hidden_c)
        self.add_pos_emb_depth = AddPosEmb(n_vecs, hidden_d)
        self.sc = SoftComp(channel_c, hidden_c, output_size, kernel_size, stride, padding)
        self.encoder = Encoder(channel_c)
        self.encoder_depth = DepthEncoder_light(channel_d)

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(channel_c, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames, depth_frames):
        # extracting features
        b, t, c, h, w = masked_frames.size()
        time0 = time.time()
        enc_feat = self.encoder(masked_frames.view(b * t, c, h, w))
        depth_frames = depth_frames.view(b * t, 1, h, w)
        enc_feat_depth = self.encoder_depth(depth_frames)
        _, c, h, w = enc_feat.size()
        # trans_feat for color channels
        trans_feat = self.ss(enc_feat, b)
        trans_feat = self.add_pos_emb(trans_feat)
        _,_,c1 = trans_feat.size()
        # trans_feat for depth channel
        trans_feat_depth = self.ss_depth(enc_feat_depth, b)
        trans_feat_depth = self.add_pos_emb_depth(trans_feat_depth)
        _,_,c2 = trans_feat_depth.size()
        # transformer body
        trans_feat, trans_feat_depth = torch.split(self.transformer1(torch.cat((trans_feat,trans_feat_depth),dim=2)), [c1,c2], dim=2)
        trans_feat = self.transformer2(trans_feat)
        trans_feat = self.sc(trans_feat, t)
        enc_feat = enc_feat + trans_feat
        pred_frames = self.decoder(enc_feat)
        pred_frames = torch.tanh(pred_frames)
        return pred_frames, enc_feat

# stage-3: content enhancement
class ContentEnhancement(BaseNetwork):
    def __init__(self,
                 dim=128,
                 spynet_path=None,
                 pa_frames=2,
                 stack_num=4,
                 deformable_groups=16,
                 max_residue_magnitude=10,
                 init_weights=True):
        super().__init__()

        self.pa_frames = pa_frames
        self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])
        blocks = []
        for _ in range(stack_num):
            blocks.append(EnhanceBlock(dim=dim,
                                       pa_frames=pa_frames,
                                       deformable_groups=deformable_groups,
                                       max_residue_magnitude=max_residue_magnitude))
        self.blocks = nn.ModuleList(blocks)

        #self.encoder = Encoder(dim)
        self.decoder = nn.Sequential(
            deconv(dim, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x, x_feat):
        b, t, c, h, w = x.shape
        # calculate flows
        x_ = F.interpolate(x.view(-1, c, h, w),
                                            scale_factor=1 / 4,
                                            mode='bilinear',
                                            align_corners=True,
                                            recompute_scale_factor=True).view(b, t, c, h//4, w//4)
        flows_backward, flows_forward = self.get_flows(x_)
        # warp input
        #x = x.view(b*t, c, h, w)
        #x_feat = self.encoder(x)
        b, t, c_, h_, w_ = x_feat.shape
        for _, block in enumerate(self.blocks):
            x_feat = x_feat + block(x_feat, flows_backward, flows_forward)
        output = self.decoder(x_feat.view(b*t, c_, h_, w_))
        output = torch.tanh(output)
        return output

    def get_flows(self, x):
        ''' Get flows for 2 frames, 4 frames or 6 frames.'''

        if self.pa_frames == 2:
            flows_backward, flows_forward = self.get_flow_2frames(x)
        elif self.pa_frames == 4:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames
            flows_forward = flows_forward_2frames + flows_forward_4frames
        elif self.pa_frames == 6:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
            flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames

        return flows_backward, flows_forward
    
    def get_aligned_image_2frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 2 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i+1 aligned towards i

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i-1 aligned towards i

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_flow_2frames(self, x):
        '''Get flow between frames t and t+1 from x.'''

        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # backward
        flows_backward = self.spynet(x_1, x_2)
        flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                          zip(flows_backward, range(4))]

        # forward
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                         zip(flows_forward, range(4))]

        return flows_backward, flows_forward

    def get_flow_4frames(self, flows_forward, flows_backward):
        '''Get flow between t and t+2 from (t,t+1) and (t+1,t+2).'''

        # backward
        d = flows_forward[0].shape[1]
        flows_backward2 = []
        for flows in flows_backward:
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
                flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+2 to i
            flows_backward2.append(torch.stack(flow_list, 1))

        # forward
        flows_forward2 = []
        for flows in flows_forward:
            flow_list = []
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
                flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-2 to i
            flows_forward2.append(torch.stack(flow_list, 1))

        return flows_backward2, flows_forward2

    def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
        '''Get flow between t and t+3 from (t,t+2) and (t+2,t+3).'''

        # backward
        d = flows_forward2[0].shape[1]
        flows_backward3 = []
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
                flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+3 to i
            flows_backward3.append(torch.stack(flow_list, 1))

        # forward
        flows_forward3 = []
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
                flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-3 to i
            flows_forward3.append(torch.stack(flow_list, 1))

        return flows_backward3, flows_forward3


# #############################################################################
# Encoder-Decoder
# #############################################################################

class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256 , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, dim, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, h, w = x.size()
        h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out) # 128
        return out

class DepthEncoder(nn.Module):
    def __init__(self, dim):
        super(DepthEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, dim , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out = self.layers(x) #32
        return out

class DepthEncoder_light(nn.Module):
    def __init__(self, dim):
        super(DepthEncoder_light, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, dim , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out = self.layers(x) 
        return out

class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# Spatio-temporal Transformer  
# #############################################################################

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value=None, m=None):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if m is not None:
            scores.masked_fill_(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        if value==None:
            return p_attn
        else:
            p_val = torch.matmul(p_attn, value)
            return p_val

class AddPosEmb(nn.Module):
    def __init__(self, n, c):
        super(AddPosEmb, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, n, c).float().normal_(mean=0, std=0.02), requires_grad=True)
        self.num_vecs = n

    def forward(self, x):
        b, n, c = x.size()
        x = x.view(b, -1, self.num_vecs, c)
        x = x + self.pos_emb
        x = x.view(b, n, c)
        return x

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding, dropout=0.1):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, b):
        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)
        feat = feat.view(b, -1, feat.size(2))
        feat = self.dropout(feat)
        return feat

class SoftComp(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.t2t = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size
        self.bias = nn.Parameter(torch.zeros((channel, h, w), dtype=torch.float32), requires_grad=True)

    def forward(self, x, t):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = self.t2t(feat) + self.bias[None]
        return feat

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, x):
        b, n, c = x.size()
        c_h = c // self.head
        key = self.key_embedding(x)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        query = self.query_embedding(x)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        value = self.value_embedding(x)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        att = self.attention(query, key, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        output = self.output_linear(att)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x

class FusionFeedForward(nn.Module):
    def __init__(self, d_model, dim_hidden=40, p=0.1, n_vecs=None, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set d_ff as a default to 1960
        kernel_size = t2t_params['kernel_size']
        hd = reduce(mul,kernel_size)*dim_hidden
        self.conv1 = nn.Sequential(
            nn.Linear(d_model, hd))
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(hd, d_model),
            nn.Dropout(p=p))
        assert t2t_params is not None and n_vecs is not None
        tp = t2t_params.copy()
        self.fold = nn.Fold(**tp)
        del tp['output_size']
        self.unfold = nn.Unfold(**tp)
        self.n_vecs = n_vecs

    def forward(self, x):
        x = self.conv1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, 49).view(-1, self.n_vecs, 49).permute(0, 2, 1)
        x = self.unfold(self.fold(x.view(-1, self.n_vecs, c).permute(0, 2, 1)) / self.fold(normalizer)).permute(0, 2,
                                                                                                                1).contiguous().view(
            b, n, c)
        x = self.conv2(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=512, dim_hidden=40, num_head=4, dropout=0.1, n_vecs=None, t2t_params=None):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout)
        self.ffn = FusionFeedForward(hidden, dim_hidden, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = self.norm1(input)
        x = input + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)
        return x

class MultiHeadedAttentionWithDepth(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d1_model, d2_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d1_model+d2_model, d1_model+d2_model)
        self.key_embedding = nn.Linear(d1_model+d2_model, d1_model+d2_model)
        self.color_value_embedding = nn.Linear(d1_model, d1_model)
        self.depth_value_embedding = nn.Linear(d2_model, d2_model)
        self.color_linear = nn.Linear(d1_model, d1_model)
        self.depth_linear = nn.Linear(d2_model, d2_model)
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, color_tokens, depth_tokens):
        b, n, c1 = color_tokens.size()
        _, _, c2 = depth_tokens.size()
        c1_h = c1 // self.head
        c2_h = c2 // self.head
        key = self.key_embedding(torch.cat((color_tokens,depth_tokens),dim=2))
        key = key.view(b, n, self.head, c1_h+c2_h).permute(0, 2, 1, 3)
        query = self.query_embedding(torch.cat((color_tokens,depth_tokens),dim=2))
        query = query.view(b, n, self.head, c1_h+c2_h).permute(0, 2, 1, 3)
        attn = self.attention(query, key, value=None)
        color_value = self.color_value_embedding(color_tokens)
        color_value = color_value.view(b, n, self.head, c1_h).permute(0, 2, 1, 3)       
        color_value = torch.matmul(attn, color_value).permute(0, 2, 1, 3).contiguous().view(b, n, c1)
        color_value = self.color_linear(color_value)
        depth_value = self.depth_value_embedding(depth_tokens)
        depth_value = depth_value.view(b, n, self.head, c2_h).permute(0, 2, 1, 3)       
        depth_value = torch.matmul(attn, depth_value).permute(0, 2, 1, 3).contiguous().view(b, n, c2)
        depth_value = self.depth_linear(depth_value)
        return color_value, depth_value

class TransformerBlockWithDepth(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden1=512, hidden2=128, dim_hidden1=40, dim_hidden2=16, num_head=4, dropout=0.1, n_vecs=None, t2t_params=None):
        super().__init__()
        self.attention = MultiHeadedAttentionWithDepth(d1_model=hidden1, d2_model=hidden2, head=num_head, p=dropout)
        self.ffn1 = FusionFeedForward(hidden1, dim_hidden=dim_hidden1, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params)
        self.ffn2 = FusionFeedForward(hidden2, dim_hidden=dim_hidden2, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params)
        self.c_norm1 = nn.LayerNorm(hidden1)
        self.c_norm2 = nn.LayerNorm(hidden1)
        self.d_norm1 = nn.LayerNorm(hidden2)
        self.d_norm2 = nn.LayerNorm(hidden2)
        #self.dropout = nn.Dropout(p=dropout)
        self.c1 = hidden1
        self.c2 = hidden2

    def forward(self, input):
        color_token, depth_token = torch.split(input,[self.c1,self.c2],dim=2)
        x1 = self.c_norm1(color_token)
        x2 = self.d_norm1(depth_token)
        #x = input + self.dropout(self.attention(x1,x2))
        x1, x2 = self.attention(x1,x2)
        x1 = color_token + x1
        x1 = x1 + self.ffn1(self.c_norm2(x1))
        x2 = depth_token + x2
        x2 = x2 + self.ffn2(self.d_norm2(x2))
        output = torch.cat((x1,x2),dim=2)
        return output

# #############################################################################
#  Flow-guided DCN  
# #############################################################################

class ModulatedDeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    # def forward(self, x, offset, mask):
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)

class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    # def forward(self, x):
    #     out = self.conv_offset(x)
    #     o1, o2, mask = torch.chunk(out, 3, dim=1)
    #     offset = torch.cat((o1, o2), dim=1)
    #     mask = torch.sigmoid(mask)
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)

class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.

    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.pa_frames = kwargs.pop('pa_frames', 2)

        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d((1+self.pa_frames//2) * self.in_channels + self.pa_frames, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.pa_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1)//2, 1, 1)
        elif self.pa_frames == 4:
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4': # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output

# Flow Estimation, SpyNet
class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)

class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x

class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2**(5-level) # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h//scale, w//scale), mode='bilinear', align_corners=False)
                flow_out[:, 0, :, :] *= float(w//scale) / float(w_floor//scale)
                flow_out[:, 1, :, :] *= float(h//scale) / float(h_floor//scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list

class EnhanceBlock(nn.Module):
    def __init__(self,
                 dim,
                 pa_frames=2,
                 deformable_groups=16,
                 max_residue_magnitude=10
                 ):
        super(EnhanceBlock, self).__init__()
        self.pa_frames = pa_frames
        # parallel warping
        self.pa_deform = DCNv2PackFlowGuided(dim, dim, 3, padding=1, deformable_groups=deformable_groups,
                                                 max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
        self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)
    
    def forward(self, x, flows_backward, flows_forward):
        '''input x: [b, t, c, h, w]'''
        x_backward, x_forward = getattr(self, f'get_aligned_feature_{self.pa_frames}frames')(x, flows_backward, flows_forward)
        # x: [b, t, c, h, w]
        x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        return x

    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 2 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 4 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n, 1, -1):
            x_i = x[:, i - 1, ...]
            flow1 = flows_backward[0][:, i - 2, ...]
            if i == n:
                x_ii = torch.zeros_like(x[:, n - 2, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_backward[1][:, i - 2, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
            x_backward.insert(0,
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(-1, n - 2):
            x_i = x[:, i + 1, ...]
            flow1 = flows_forward[0][:, i + 1, ...]
            if i == -1:
                x_ii = torch.zeros_like(x[:, 1, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_forward[1][:, i, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
            x_forward.append(
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 6 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n + 1, 2, -1):
            x_i = x[:, i - 2, ...]
            flow1 = flows_backward[0][:, i - 3, ...]
            if i == n + 1:
                x_ii = torch.zeros_like(x[:, -1, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            elif i == n:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = x[:, i, ...]
                flow3 = flows_backward[2][:, i - 3, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i+3 aligned towards i
            x_backward.insert(0,
                              self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                             x[:, i - 3, ...], [flow1, flow2, flow3]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow1 = flows_forward[0][:, i, ...]
            if i == 0:
                x_ii = torch.zeros_like(x[:, 0, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            elif i == 1:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = x[:, i - 2, ...]
                flow3 = flows_forward[2][:, i - 2, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i-3 aligned towards i
            x_forward.append(self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                            x[:, i + 1, ...], [flow1, flow2, flow3]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

   
# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################

class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels, out_channels=nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module