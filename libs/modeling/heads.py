import math
import time
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import DTFAM, MaskedConv1D, Scale, LayerNorm, DynamicScale_chk


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits

class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )

        # fpn_masks remains the same
        return out_offsets


class TDynPtTransformerClsHead(nn.Module):
    """
    Shared 1D MSDy-head for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=5,
        act_layer=nn.ReLU,
        empty_cls = [],
        gate_activation_kargs: dict = None
    ):
        super().__init__()
        self.act = act_layer()
        
        assert num_layers-1 >0

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()

        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            cls_subnet_conv = DTFAM(dim=in_dim, o_dim= feat_dim, ka=kernel_size, conv_type = 'others', gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs)

            self.head.append(
                DynamicScale_chk(
                in_dim,
                out_dim,
                num_convs=1,
                kernel_size=kernel_size,
                padding=1,
                stride=kernel_size // 2,
                num_groups=1,
                num_adjacent_scales=2,
                depth_module=cls_subnet_conv,
                gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs))

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        feats = fpn_feats
        for i in range(len(self.head)):
            feats, fpn_masks = self.head[i](feats, fpn_masks)
            
            for j in range(len(feats)):
                feats[j] =  self.act(feats[j])

        for cur_out, cur_mask in zip(feats, fpn_masks):
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        return out_logits

class TDynPtTransformerRegHead(nn.Module):
    """
    Shared 1D MSDy-head for regression
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=5,
        act_layer=nn.ReLU,
        gate_activation_kargs: dict = None,
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        
        assert num_layers-1 > 0

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            reg_subnet_conv = DTFAM(dim=in_dim, o_dim= feat_dim, ka=kernel_size, conv_type = 'others', gate_activation=gate_activation_kargs["type"],
                gate_activation_kargs = gate_activation_kargs)

            self.head.append(DynamicScale_chk(
                in_dim,
                out_dim,
                num_convs=1,
                kernel_size=kernel_size,
                padding=1,
                stride=kernel_size // 2,
                num_groups=1,
                num_adjacent_scales=2,
                depth_module=reg_subnet_conv,
                gate_activation=gate_activation_kargs['type'],
                gate_activation_kargs = gate_activation_kargs)
                )

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )


    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels
        
        # apply the classifier for each pyramid level
        out_offsets = tuple()
        feats = fpn_feats
        for i in range(len(self.head)):
            
            feats, fpn_masks = self.head[i](feats, fpn_masks)
            for j in range(len(feats)):
                feats[j] = self.act(feats[j])
        
        for l in range(self.fpn_levels):
            cur_offsets, _  = self.offset_head(feats[l], fpn_masks[l])            
            out_offsets +=  ( F.relu(self.scale[l](cur_offsets)) , )
               
        return out_offsets 
