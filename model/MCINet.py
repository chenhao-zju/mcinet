r""" Multi-Content Interaction Network for Few-Shot Segmentation """
from builtins import int
from functools import reduce
from operator import add
from xml.etree.ElementTree import TreeBuilder
from einops import rearrange

import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from timm.models.layers import DropPath


from .base.swin_transformer import SwinTransformer
from .base.transformer import MultiHeadedAttention, PositionalEncoding, MultiHeadAttentionOne
from .base.conv4d import make_building_block
from .base.aspp import ASPP


# from base.swin_transformer import SwinTransformer
# from base.transformer import MultiHeadedAttention, PositionalEncoding, MultiHeadedAttentionMix, MultiHeadAttentionOne, PoolingAttention
# from base.conv4d import make_building_block
# from base.aspp import ASPP


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def mid_mixer_conv(in_channel, mid_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, mid_channel, (3, 3), padding=(1, 1), bias=True),
                        nn.ReLU(),
                        nn.Conv2d(mid_channel, out_channel, (3, 3), padding=(1, 1), bias=True),
                        nn.ReLU())

def detect_head_conv(in_channel, mid_channel):
    return nn.Sequential(nn.Conv2d(in_channel, mid_channel, (3, 3), padding=(1, 1), bias=True),
                        nn.ReLU(),
                        nn.Conv2d(mid_channel, 2, (3, 3), padding=(1, 1), bias=True))


class convnext(nn.Module):
    r""" 
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channel, mid_channel, out_channel, res=False, drop_path=0., layer_scale_init_value=1e-6):
        super(convnext, self).__init__()
        self.res = res
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel) # depthwise conv
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, mid_channel) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mid_channel, out_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        if self.res:
            x = input + self.drop_path(x)
        else:
            x = self.drop_path(x)
        return x

class MCINet(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize, original=True, cross_mix=False,
                        add_low=False, add_4dconv=False, use_aspp=False, upmix=False, skip_mode='concat', 
                        pooling_mix='concat', mixing_mode='concat', mix_out='mixer3', combine_mode='add', model_mask=[1,2,3]):
        super(MCINet, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize
        self.original = original
        self.cross_mix = cross_mix
        self.add_low = add_low
        self.add_4dconv = add_4dconv
        self.use_aspp = use_aspp
        self.upmix = upmix
        self.skip_mode = skip_mode
        self.pooling_mix = pooling_mix
        self.mixing_mode = mixing_mode
        self.mix_out = mix_out
        self.combine_mode = combine_mode
        self.model_mask = model_mask

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)          # self.nlayers = [a, b, c, d] --> [a, a+b, a+b+c, a+b+c+d]
        self.model = MCINet_model(in_channels=self.feat_channels, stack_ids=self.stack_ids, original=self.original, mix_out=self.mix_out, cross_mix=self.cross_mix, add_low=self.add_low, 
                                 add_4dconv=self.add_4dconv, use_aspp=self.use_aspp, upmix=self.upmix, skip_mode=self.skip_mode, pooling_mix=self.pooling_mix, mixing_mode=self.mixing_mode, 
                                 combine_mode=self.combine_mode, model_mask=self.model_mask)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, query_masks, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)

        logit_mask = self.model(query_img, query_feats, query_masks, support_img, support_feats, support_mask.clone())

        return logit_mask

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        query_mask = batch['query_mask']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            logit_mask = self(query_img, query_mask, support_imgs.squeeze(1), support_masks.squeeze(1))
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)
            logit_mask = self.model(query_img, query_feats, query_mask, support_imgs, n_support_feats, support_masks.clone(), nshot)

        return logit_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()


class MCINet_model(nn.Module):
    def __init__(self, in_channels, stack_ids, original=False, cross_mix=False, add_low=False, 
                    add_4dconv=False, use_aspp=False, upmix=False, skip_mode='mix', pooling_mix='concat', 
                    mixing_mode='concat', mix_out='mixer3', combine_mode='add', model_mask=[1,2,3]):
        super(MCINet_model, self).__init__()

        self.model_mask = model_mask

        self.stack_ids = stack_ids
        self.original = original
        self.cross_mix = cross_mix
        self.add_low = add_low
        self.add_4dconv = add_4dconv
        self.use_aspp = use_aspp
        self.upmix = upmix
        self.skip_mode = skip_mode
        self.pooling_mix = pooling_mix
        self.mixing_mode = mixing_mode
        self.mix_out = mix_out
        self.combine_mode = combine_mode

        outch1, outch2, outch3 = 16, 64, 128
        feature_dim = [96, 48, 24, 12]

        self.head = 4

        # MCINet blocks
        if self.add_4dconv:
            self.linears = nn.ModuleDict()
            self.conv_4d_blocks = nn.ModuleDict()
        else:
            self.MCINet_blocks = nn.ModuleDict()
        
        self.pe = nn.ModuleDict()
        for idx in self.model_mask:
            inch = in_channels[idx]
            if self.add_4dconv:
                print('add 4D conv')
                if self.cross_mix:
                    in_channel = stack_ids[idx]-stack_ids[idx-1]+1
                else: 
                    in_channel = stack_ids[idx]-stack_ids[idx-1]
                self.linears[str(idx)] = clones(nn.Linear(inch, inch), 2)
                self.conv_4d_blocks[str(idx)] = make_building_block(in_channel, [outch1, outch1], kernel_sizes=[5,3], spt_strides=[1,1], type='6dconv')
            else:
                print('original')
                self.MCINet_blocks[str(idx)] = MultiHeadedAttention(h=8, d_model=inch, dropout=0.5, add_gaussian=self.add_gaussian, add_bottle_layer=self.add_bottle_layer)
           
            self.pe[str(idx)] = PositionalEncoding(d_model=inch, dropout=0.5)


        # cross_layer mixing attention
        if self.cross_mix:
            if self.cross_mix: print('cross layer mixing')
            self.cross_bottlenecks = nn.ModuleDict()
            for idx in self.model_mask:
                self.cross_bottlenecks[str(idx)] = nn.Sequential(nn.Conv2d(in_channels[idx-1], in_channels[idx], (3, 3), padding=(1, 1), bias=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels[idx], in_channels[idx], (1, 1), bias=True),
                                                                    nn.ReLU())
                

        if self.add_low:
            print('add low image feature')
            self.low_cross_blocks = nn.ModuleDict()
            self.low_bottlenecks = nn.ModuleDict()

            feature_dim = [96, 48, 24, 12]
            self.low_feat_layer = self.model_mask[:]

            for idx in self.low_feat_layer:
                self.low_cross_blocks[str(idx)] = MultiHeadAttentionOne(n_head=1, d_model=int(feature_dim[idx]**2), d_value=int(feature_dim[idx]**2), dropout=0.5)
                self.low_bottlenecks[str(idx)] = nn.Sequential(nn.Conv2d(in_channels[1], 128, (3, 3), padding=(1, 1), bias=True),
                                                                nn.ReLU(),
                                                                nn.Conv2d(128, 128, (1, 1), bias=True),
                                                                nn.ReLU())


        # conv blocks
        self.conv_layer = nn.ModuleDict()
        if self.add_loss:
            print('add_loss')
            self.mid_loss_heads = nn.ModuleDict()
            self.loss_functions = {}

        self.key = self.head if self.add_pool4d else 1
        for idx in self.model_mask:
            in_channel = self.key*(stack_ids[idx]-stack_ids[idx-1]) if idx>0 else self.key*stack_ids[idx]
            if self.cross_mix: in_channel = in_channel+1 
            
            if idx == 2: z = idx+1
            elif idx == 3: z = idx-1
            
            if self.add_4dconv: in_channel = outch1
            

            self.conv_layer[str(idx)] = self.build_conv_block(in_channel, [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
            

        if self.upmix:
            print('use_upmix')
            if 3 in self.model_mask:
                if 1 in self.model_mask or 2 in self.model_mask:
                    self.feature_mixing32 = self.build_conv_block(2*outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
            if 2 in self.model_mask and 1 in self.model_mask:
                self.feature_mixing21 = self.build_conv_block(2*outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
            self.mid_mixer = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

            in_channel = outch3+2*in_channels[1]+2*in_channels[0]
            self.mid_mixer1 = mid_mixer_conv(in_channel, outch3, outch2)
            self.mid_mixer2 = mid_mixer_conv(outch2, outch2, outch1)
            self.classify_head = detect_head_conv(outch1, outch1)
            self.loss_function = nn.CrossEntropyLoss()

        if self.combine_mode == 'add':
            print('add')
            in_channel4 = in_channel5 = in_channel6 = outch3
        elif self.combine_mode == 'concat':
            print('concat')
            in_channel4 = 2*outch3 if 1 in self.model_mask else outch3
            in_channel5 = 2*outch3 if 2 in self.model_mask else outch3
            in_channel6 = 2*outch3 if 3 in self.model_mask else outch3

        # name of layer is 4-6 from top to bottom
        if 2 in self.model_mask or 3 in self.model_mask:
            self.conv4 = self.build_conv_block(in_channel4, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        if 1 in self.model_mask or 2 in self.model_mask:
            self.conv5 = self.build_conv_block(in_channel5, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8
        self.conv6 = self.build_conv_block(in_channel6, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8


        # mixer blocks
        in_channel = 2*outch3+in_channels[0] if self.use_fpn else outch3+2*in_channels[1]+2*in_channels[0]
        in_channel = in_channel if self.mixing_mode=='concat' else outch3
        in_channel = outch3+2*in_channels[0] if self.use_convnext else in_channel
        if self.use_aspp:
            print('use_aspp')
            # in_channel = outch3+2*in_channels[0]
            # if self.skip_mode == 'concat':
            #     in_channel += 1
            if self.upmix:
                in_channel += 1
            self.mid_neck = mid_mixer_conv(in_channel, outch3, outch3)
            self.aspp = ASPP(outch3)
            in_channel = 5*outch3

        print('mixer_output1')
        self.mixer1 = detect_head_conv(in_channel, outch3) if self.mix_out=='mixer1' else mid_mixer_conv(in_channel, outch3, outch2)
        if self.mix_out=='mixer2' or self.mix_out=='mixer3':
            print('mixer_output2')
            self.mixer2 = detect_head_conv(outch2, outch2) if self.mix_out=='mixer2' else mid_mixer_conv(outch2, outch2, outch1)
        if self.mix_out=='mixer3':
            print('mixer_output3')
            self.mixer3 = detect_head_conv(outch1, outch1)


    def forward(self, query_img, query_feats, query_masks, support_imgs, support_feats, support_mask, nshot=1):
        coarse_masks = {i:[] for i in self.model_mask}
        target_masks = {}
        support_masks = {}
        coarse_similaritys = {i:[] for i in self.model_mask}
        low_query_feats = {}
        low_support_feats = {}
        mid_results = {}
        mid_loss = 0
        upsample_times = 2
        
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx<self.stack_ids[0]: 
                if 0 not in self.model_mask: continue
                else: key = '0'
            if idx<self.stack_ids[1] and idx>=self.stack_ids[0]: 
                if 1 not in self.model_mask: continue
                else: key = '1'
            if idx<self.stack_ids[2] and idx>=self.stack_ids[1]: 
                if 2 not in self.model_mask: continue
                else: key = '2'
            if idx<self.stack_ids[3] and idx>=self.stack_ids[2]: 
                if 3 not in self.model_mask: continue
                else: key = '3'

            bsz, ch, ha, wa = query_feat.size()
            query_mask = F.interpolate(query_masks.unsqueeze(1).float(), (ha, wa), mode='bilinear',
                                    align_corners=True)
            q_mask = query_mask.view(bsz, -1).unsqueeze(-1)

            # reshape the input feature and mask
            query_feat_reshape = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            if nshot == 1:
                support_feat = support_feats[idx]
                s_mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                    align_corners=True)
                mask = s_mask.view(support_feat.size()[0], -1)
                support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                # support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                support_feat = rearrange(support_feat, 'n b c h w -> (n b) (h w) c').contiguous()
                s_mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask], dim=1)
                # mask = s_mask.view(bsz, -1)
                mask = rearrange(s_mask, 'n b c h w -> b (n c h w)')

            if self.add_low:
                query_input = F.interpolate(query_feats[self.stack_ids[1]-1], (ha, wa), mode='bilinear', align_corners=True)
                query_input_feat = self.low_bottlenecks[key](query_input)

                query_feat = rearrange(query_feat, 'b c h w -> b c (h w)')
                query_input_feat = rearrange(query_input_feat, 'b c h w -> b c (h w)')

                low_query_feat = self.low_cross_blocks[key](q=query_feat, k=query_input_feat, v=query_input_feat, residual=query_feat).transpose(-1, -2)

                if idx+1 in self.stack_ids:
                    low_query_feats[int(key)] = rearrange(low_query_feat, 'b (h w) c -> b c h w', h=ha, w=wa)
                    # low_support_feats[int(key)] = rearrange(low_support_feat, 'b (h w) c->b c h w', h=ha, w=wa)

            # positioned embedding
            query = self.pe[key](low_query_feat) if self.add_low else self.pe[key](query_feat_reshape)
            support = self.pe[key](support_feat)
            
            if self.add_4dconv:
                query, support = [l(x) for l, x in zip(self.linears[key], (query, support))]
                query = rearrange(query, 'b p (head c) -> b head p c', head=self.head)
                support = rearrange(support, '(n b) p (head c) -> b head (n p) c', n=nshot, head=self.head)
                similarity = query@support.transpose(-1, -2)
                # print(similarity.shape)
                similarity = rearrange(similarity, 'b head (h1 w1) (n h2 w2) -> (n head b) h1 w1 h2 w2', h1=ha, w1=wa, n=nshot, h2=ha, w2=wa)
                # print(similarity.shape)
                coarse_similaritys[int(key)].append(similarity)
                support_masks[int(key)] = mask
            else:
                coarse_mask = self.MCINet_blocks[key](query, support, mask, nshot=nshot)
                coarse_masks[int(key)].append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, ha, wa))

            if idx+1 in self.stack_ids:
                target_masks[int(key)] = query_mask.squeeze(1)


        # cross layer mixing
        if self.cross_mix:
            cross_mix_feature = {}
            for idx in self.model_mask:
                key = str(idx)

                if nshot == 1:
                    support_feat = support_feats[self.stack_ids[idx]-1]
                    ha, wa = support_feat.size()[-2:]
                    s_mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                        align_corners=True)
                    mask = s_mask.view(support_feat.size()[0], -1)
                    support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
                else:
                    support_feat = torch.stack([support_feats[k][self.stack_ids[idx] - 1] for k in range(nshot)])
                    ha, wa = support_feat.size()[-2:]
                    # support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                    support_feat = rearrange(support_feat, 'n b c h w -> (n b) (h w) c').contiguous()
                    s_mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                        for k in support_mask], dim=1)
                    # mask = s_mask.view(bsz, -1)
                    mask = rearrange(s_mask, 'n b c h w -> b (n c h w)')

                support = self.pe[key](support_feat)

                # reshape the input feature and mask
                query_feat = query_feats[self.stack_ids[idx-1] - 1]
                if query_feat.size()[-1] != wa:
                    query_feat = self.cross_bottlenecks[key](query_feat)
                    query_feat = F.interpolate(query_feat, (ha, wa), mode='bilinear', align_corners=True)
                bsz, ch, _, _ = query_feat.shape
                query_feat_reshape = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()

                # print(query_feat_reshape.shape)
                # print(support_feat.shape)
                # print(mask.shape)
                query = self.pe[key](query_feat_reshape)

                if self.add_4dconv:
                    query, support = [l(x) for l, x in zip(self.linears[key], (query, support))]
                    query = rearrange(query, 'b p (head c) -> b head p c', head=self.head)
                    support = rearrange(support, '(n b) p (head c) -> b head (n p) c', n=nshot, head=self.head)
                    similarity = query@support.transpose(-1, -2)
                    # print(similarity.shape)
                    similarity = rearrange(similarity, 'b head (h1 w1) (n h2 w2) -> (n head b) h1 w1 h2 w2', h1=ha, w1=wa, n=nshot, h2=ha, w2=wa)
                    # print(similarity.shape)
                    coarse_similaritys[int(key)].append(similarity)
              
                else:
                    coarse_mask = self.MCINet_blocks[key](query, support, mask, nshot=nshot)
                    # print(coarse_mask.shape)
                    cross_mix_feature[idx] = coarse_mask.contiguous().view(bsz, ha, wa)

        
        if self.add_4dconv:
            coarse_masks = {}
            for idx in self.model_mask:
                bsz, ch, ha, wa = query_feats[self.stack_ids[idx] - 1].size()
                mix_similarity = torch.stack(coarse_similaritys[idx], dim=1)
                # print(mix_similarity.shape)
                mix_similarity = self.conv_4d_blocks[str(idx)](mix_similarity)
                mix_similarity = rearrange(mix_similarity, '(n head b) t h1 w1 h2 w2 -> b (head t) (h1 w1) (n h2 w2)', n=nshot, head=self.head)
                mix_similarity = F.softmax(mix_similarity, dim=-1)

                num_channel = mix_similarity.shape[1]
                support_mask0 = support_masks[idx]
                support_mask0 = support_mask0.repeat(num_channel, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
                
                coarse_mask = torch.matmul(mix_similarity, support_mask0)
                coarse_mask = rearrange(coarse_mask, 'b (head t) (h w) n -> (b n) head t h w', h=ha, w=wa, head=self.head)
                coarse_mask = torch.mean(coarse_mask, 1)

                coarse_masks[idx] = coarse_mask
        

        coarse_masks0 = {}
        for key, mask_list in coarse_masks.items():
            if self.add_4dconv or self.add_pool4d:
                value = mask_list.clone()
            else:
                bsz, ha, wa = mask_list[0].size()
                if self.cross_mix:
                    mask_list.append(cross_mix_feature[key])
                value = torch.stack(mask_list, dim=1).contiguous()
                # print(value.shape)

            if self.combine_mode == 'add':
                predict_similarity = self.mid_loss_heads[str(key)](value)
                mid_loss += self.loss_functions[key](predict_similarity, target_masks[key].long())

            coarse_masks0[key] = self.conv_layer[str(key)](value)
        

        # multi-scale cascade (pixel-wise addition)
        if self.combine_mode == 'concat':
            mid_loss += self.loss_functions[3](coarse_masks0[3], target_masks[3].long())

        mix_masks = {}
        if 3 in self.model_mask:
            coarse_masks3 = coarse_masks0[3]
            if self.upmix:
                if 2 in self.model_mask:
                    coarse_masks2 = coarse_masks0[2]
                    mid_feature = F.interpolate(coarse_masks2, coarse_masks3.size()[2:], mode='bilinear', align_corners=True)
                else:
                    if 1 in self.model_mask:
                        coarse_masks1 = coarse_masks0[1]
                        mid_feature = F.interpolate(coarse_masks1, coarse_masks3.size()[2:], mode='bilinear', align_corners=True) 
                coarse_masks3 = torch.cat([mid_feature, coarse_masks3], dim=1)
                coarse_masks3 = self.feature_mixing32(coarse_masks3)
            bsz, ch, ha, wa = coarse_masks3.size()
            coarse_masks3 = F.interpolate(coarse_masks3, (upsample_times*ha, upsample_times*wa), mode='bilinear', align_corners=True)
            mix_masks[3] = coarse_masks3
            if 2 in self.model_mask:
                if self.combine_mode == 'add':
                    mix = coarse_masks3 + coarse_masks0[2]
                elif self.combine_mode == 'concat':
                    mix = torch.cat([coarse_masks3, coarse_masks0[2]], dim=1)
                # print('3 and 2 mixing')
            else:
                mix = coarse_masks3
            mix = self.conv4(mix)

        if self.combine_mode == 'concat':
            mid_loss += self.loss_functions[2](mix, target_masks[2].long())

        if 2 in self.model_mask or 3 in self.model_mask:
            coarse_masks2 = mix if 3 in self.model_mask else coarse_masks0[2]
            if self.upmix:
                if 1 in self.model_mask:
                    coarse_masks1 = coarse_masks0[1]
                    mid_feature = F.interpolate(coarse_masks1, coarse_masks2.size()[2:], mode='bilinear', align_corners=True) 
                    coarse_masks2 = torch.cat([mid_feature, coarse_masks2], dim=1)
                    coarse_masks2 = self.feature_mixing21(coarse_masks2)
            bsz, ch, ha, wa = coarse_masks2.size()
            coarse_masks2 = F.interpolate(coarse_masks2, (upsample_times*ha, upsample_times*wa), mode='bilinear', align_corners=True)
            mix_masks[2] = coarse_masks2
            # print('2_upsample')
            if 1 in self.model_mask:
                if self.combine_mode == 'add':
                    mix = coarse_masks2 + coarse_masks0[1]
                elif self.combine_mode == 'concat':
                    mix = torch.cat([coarse_masks2, coarse_masks0[1]], dim=1)
                # print('2 and 1 mixing')
            else:
                mix = coarse_masks2

            mix = self.conv5(mix)
        elif 1 in self.model_mask:
            mix = coarse_masks0[1]
            mix = self.conv5(mix)
            # print('output1')
        
        if 1 in self.model_mask:
            bsz, ch, ha, wa = mix.size()
            mix0 = F.interpolate(mix, (upsample_times*ha, upsample_times*wa), mode='bilinear', align_corners=True)
            mix_masks[1] = mix0

        if self.combine_mode == 'concat':
            mid_loss += self.loss_functions[1](mix, target_masks[1].long())


        # skip connect 1/8 and 1/4 features (concatenation)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]
            if self.skip_mode == 'mix':
                s_mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                align_corners=True)
                s_mask = s_mask.expand(support_feat.shape)
                support_feat = support_feat * s_mask

        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)])
            if self.skip_mode == 'mix':
                s_mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), support_feat.size()[3:], mode='bilinear', align_corners=True)
                                    for k in support_mask], dim=1)
                s_mask = s_mask.expand(support_feat.shape)
                support_feat = support_feat * s_mask
                support_feat = support_feat.max(dim=0).values
            else:
                support_feat = support_feat.max(dim=0).values
            
        query_feat = low_query_feats[1] if self.new_skip else query_feats[self.stack_ids[1] - 1]
        
        if self.upmix:
            q_coarse_mask = mix_masks[3]
            # print(q_coarse_mask.shape)
            q_coarse_mask = self.mid_mixer(q_coarse_mask)
            q_coarse_mask = F.interpolate(q_coarse_mask, query_feat.size()[2:], mode='bilinear', align_corners=True)
            q_coarse_mask = torch.cat([q_coarse_mask, query_feat, support_feat], dim=1)
            # print(q_coarse_mask.shape)

        # print(query_feat.shape)
        mix = torch.cat((mix, query_feat, support_feat), 1)

        upsample_size = (mix.size(-1) * upsample_times,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)

        if nshot == 1:
            support_feat = support_feats[self.stack_ids[0] - 1]
            if self.skip_mode == 'mix':
                s_mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                align_corners=True)
                s_mask = s_mask.expand(support_feat.shape)
                support_feat = support_feat * s_mask

        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)])
            if self.skip_mode == 'mix':
                s_mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), support_feat.size()[3:], mode='bilinear', align_corners=True)
                                    for k in support_mask], dim=1)
                s_mask = s_mask.expand(support_feat.shape)
                support_feat = support_feat * s_mask
                support_feat = support_feat.max(dim=0).values
            else:
                support_feat = support_feat.max(dim=0).values

        query_feat = low_query_feats[0] if self.new_skip else query_feats[self.stack_ids[0] - 1]
        
        if self.upmix:
            q_coarse_mask = F.interpolate(q_coarse_mask, upsample_size, mode='bilinear', align_corners=True)
            # print(q_coarse_mask.shape)
            q_coarse_mask = torch.cat([q_coarse_mask, query_feat, support_feat], dim=1)
            # print(q_coarse_mask.shape)
            q_coarse_mask = self.mid_mixer1(q_coarse_mask)
            q_coarse_mask = self.mid_mixer2(q_coarse_mask)
            q_coarse_mask = self.classify_head(q_coarse_mask)
            # print(q_coarse_mask.shape)
            
            q_mask = F.interpolate(query_masks.unsqueeze(1).float(), upsample_size, mode='bilinear', align_corners=True)
            mid_results['gt_mask2'] = q_mask.squeeze(1)

            mid_loss += self.loss_function(q_coarse_mask, q_mask.squeeze(1).long())

            q_coarse_target = q_coarse_mask.argmax(1).unsqueeze(1)
            mid_results['pred_mask2'] = q_coarse_target.squeeze(1)

        mix = torch.cat((mix, query_feat, support_feat), 1)
        if self.upmix:
            mix = torch.cat([mix, q_coarse_target], dim=1)
        if self.use_aspp:
            mix = self.mid_neck(mix)
            mix = self.aspp(mix)


        out = self.mixer1(mix)
        upsample_size = (out.size(-1)*2*upsample_times,)*2 if self.mix_out=='mixer1' else (out.size(-1)*upsample_times,)*2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        if self.mix_out=='mixer2' or self.mix_out=='mixer3':
            out = self.mixer2(out)
            upsample_size = (out.size(-1) * upsample_times,) * 2
            out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        if self.mix_out=='mixer3':
            out = self.mixer3(out)
        
        logit_mask = out.clone()

        if self.upmix:
            return logit_mask, mid_loss, mid_results
        else:
            return logit_mask

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)



if __name__ == '__main__':
    query_image = torch.randn(1, 3, 384, 384).to('cuda:4')
    support_image = torch.randn(1, 3, 384, 384).to('cuda:4')
    support_mask = torch.ones(1, 384, 384).to('cuda:4')

    src = './model/MCINet/resnet50_a1h-35c100f8.pth'

    model = MCINet(backbone='resnet50', pretrained_path=src, use_original_imgsize=False, original=True, cross_mix=True, 
                add_low=True, add_4dconv=True, use_aspp=True, upmix=True, skip_mode='mix', pooling_mix='concat', mixing_mode='concat', 
                mix_out='mixer3', combine_mode='add', model_mask=[2,3]).to('cuda:4')

    # from torchsummaryX import summary
    # summary(model, query_image)

    outputs = model(query_image, support_mask, support_image, support_mask)

    # print(outputs.shape)
    print(outputs[0].shape)
    print(outputs[1])

