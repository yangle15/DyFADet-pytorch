import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .weight_init import trunc_normal_
except:
    from weight_init import trunc_normal_

class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        
        return out_conv, out_mask

class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out

# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale

# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
 
class DynELayer(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used in mlp,
            init_conv_vars=0.1  # init gaussian variance for the weight
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = n_ds_stride
        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi    = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convw  = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)

        
        self.fc        = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        if n_ds_stride > 1:
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):

        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)

        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))

        convw = self.convw(out)
        convkw = self.convkw(out)

        out = fc * phi + torch.relu(convw + convkw) * psi + out
        
        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()
    
class DynamicConv1D_chk(nn.Module):
    def __init__(
        self, 
        in_channels : int,
        out_channels : int,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        norm: str = "LN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None
    ):
        super(DynamicConv1D_chk, self).__init__()
        
        self.num_groups = num_groups
        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        convs = []

        convs += [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, groups= num_groups),
                    LayerNorm(out_channels)]
        in_channels = out_channels

        self.convs = nn.Sequential(*convs)
        self.gate = TemporalGate(self.in_channels,
                                #  out_dim=num_groups,
                                num_groups=num_groups,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                gate_activation=gate_activation,
                                gate_activation_kargs = gate_activation_kargs)

    def get_running_cost(self, gate):
        
        conv_cost = self.in_channels * self.out_channels * len(self.convs) * \
                self.kernel_size
        norm_cost = self.out_channels if self.norm != "none" else 0
        unit_cost = conv_cost + norm_cost

        hard_gate = (gate != 0).float()
        cost = [gate.detach() * unit_cost / self.num_groups,
                hard_gate * unit_cost / self.num_groups,
                torch.ones_like(gate) * unit_cost / self.num_groups]

        cost = [x.flatten(1).sum(-1) for x in cost]
        
        # print(cost[0]/cost[2], cost[1]/cost[2])
        
        return cost

    def forward(self, input, mask):

        out_mask = mask.to(input.dtype)
        data = self.convs(input)
        data = data * out_mask.detach()
        output = self.gate(data, input, out_mask)
        # masking the output, stop grad to mask
        output = output * out_mask.detach()
        out_mask = out_mask.bool()

        return output, out_mask

class DynamicScale_chk(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        num_convs: int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        norm: str = "GN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None
    ):
        super(DynamicScale_chk, self).__init__()
        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        dynamic_convs = [
            DTFAM(dim=in_channels, o_dim= in_channels, ka=kernel_size, gate_activation="GeReTanH",
                gate_activation_kargs = gate_activation_kargs)
            
 
        for _ in range(num_adjacent_scales)]
        
        self.dynamic_convs = nn.ModuleList(dynamic_convs)
        
        self.resize = lambda x, s : F.interpolate(
            x, size=s, mode="nearest")
        
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        self.init_parameters()

        self.norm = LayerNorm(out_channels)
        self.act = nn.ReLU()

    def init_parameters(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def forward(self, inputs, fpn_masks):

        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x, fpn_masks[l])[0] for m in self.dynamic_convs])
        
        outputs = []
        out_masks = []
        for l, x in enumerate(inputs):
            scale_feature = []
            
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-1:])
                    scale_feature.append(feature)
                         
            scale_feature = sum(scale_feature) * self.scale_weight + x * self.output_weight
            
            if self.depth_module is not None:
                scale_feature, masks = self.depth_module(scale_feature, fpn_masks[l])

            outputs.append(scale_feature)
                 
        out_masks = fpn_masks
        
        return outputs, out_masks

class TemporalGate(nn.Module):
    def __init__(
        self,
        in_channels : int,
        num_groups : int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        attn_gate : bool = False,
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None,
        head_gate = True
    ):
        super(TemporalGate, self).__init__()
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.head_gate = head_gate
        self.ka = kernel_size
        
        if num_groups == kernel_size:
            self.gate_conv = nn.Conv1d(in_channels=in_channels, out_channels=num_groups, kernel_size=kernel_size,
                           stride=stride, padding=padding)
        elif num_groups == kernel_size*in_channels:
            self.gate_conv = nn.Conv1d(in_channels=in_channels, out_channels=num_groups, kernel_size=kernel_size,
                           stride=stride, padding=padding, groups=in_channels)
        else:
            self.gate_conv = nn.Conv1d(in_channels=in_channels, out_channels=num_groups, kernel_size=kernel_size,
                           stride=stride, padding=padding, groups=num_groups)
        
        self.gate_activation = gate_activation
        self.gate_activation_kargs = gate_activation_kargs
        if gate_activation == "ReTanH":
            self.gate_activate = lambda x : torch.tanh(x).clamp(min=0)

        elif gate_activation == "ReLU":
            self.gate_activate = lambda x : torch.relu(x)

        elif gate_activation == "Sigmoid":
            self.gate_activate = lambda x : torch.sigmoid(x)

        elif gate_activation == "GeReTanH":
            assert "tau" in gate_activation_kargs
            tau = gate_activation_kargs["tau"]
            ttau = math.tanh(tau)
            self.gate_activate = lambda x : ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)
        else:
            raise NotImplementedError()

    def encode(self, *inputs):

        if self.num_groups == self.ka * self.in_channels:
            return inputs
        
        if self.num_groups == self.ka:
            da, mask = inputs
            b,ck,t = inputs[0].shape
            x = inputs[0].view(b, self.in_channels, self.ka, t)
            da = x.permute(0,2,1,3).contiguous().view(b, ck, t)
            inputs = (da,mask)
        
        outputs = [x.view(x.shape[0] * self.num_groups, -1, *x.shape[2:]) for x in inputs]

        return outputs

    def decode(self, *inputs):
        if self.num_groups == self.ka * self.in_channels:
            return inputs

        outputs = [x.view(x.shape[0] // self.num_groups, -1, *x.shape[2:]) for x in inputs]
        return outputs

    def forward(self, data_input, gate_input, mask):
        # data_input b c h w

        out_mask = mask.to(data_input.dtype)
        
        data = data_input * out_mask.detach()
        gate = self.gate_conv(gate_input)
        gate = self.gate_activate(gate)
        gate = gate*out_mask

        data, gate = self.encode(data_input, gate)
        output, = self.decode(data * gate)
        return output

class DTFAM(nn.Module):    
    def __init__(self, dim= 512, o_dim = 1, ka=3, stride=1, groups = 1, padding_mode='zeros', conv_type= 'gate', gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None):
        super().__init__()
        
        self.dim = dim

        self.padding_mode = padding_mode
        
        self.ka = ka
        self.stride = stride

        self.shift_conv = nn.Conv1d(dim, dim*ka, kernel_size=self.ka, stride= stride, bias=False, groups= dim, padding=self.ka//2, padding_mode=padding_mode)
        self.conv = nn.Conv1d(dim*ka, o_dim, kernel_size=1, bias=True, groups = groups, padding=0)

        dyn_type = gate_activation_kargs['dyn_type']
        self.conv_type = conv_type
        if self.conv_type == 'gate':
            if dyn_type == 'c':
                self.kernel_conv = TemporalGate(dim,
                                num_groups=dim,
                                kernel_size=ka,
                                padding=ka//2,
                                stride=1,
                                gate_activation=gate_activation,
                                gate_activation_kargs = gate_activation_kargs)
            elif dyn_type == 'k':
                self.kernel_conv = TemporalGate(dim,
                                num_groups=ka,
                                kernel_size=ka,
                                padding=ka//2,
                                stride=1,
                                gate_activation=gate_activation,
                                gate_activation_kargs = gate_activation_kargs)
            
            elif dyn_type == 'ck':
                self.kernel_conv = TemporalGate(dim,
                                num_groups=dim*ka,
                                kernel_size=ka,
                                padding=ka//2,
                                stride=1,
                                gate_activation=gate_activation,
                                gate_activation_kargs = gate_activation_kargs)
            else:
                assert 1==0
        else:
            self.kernel_conv = DynamicConv1D_chk(
            in_channels = dim*self.ka,
            out_channels = dim,
            kernel_size=self.ka,
            padding=self.ka//2,
            stride=stride,
            num_groups=groups,
            gate_activation=gate_activation,
            gate_activation_kargs=gate_activation_kargs)

        self.norm = LayerNorm(o_dim)
        
        self.init_parameters()
        

    def shift(self, x):
        # Pure shift operation, we do not use this operation in this repo.
        # We use constant kernel conv for shift.
        B, C, T = x.shape
        
        out = torch.zeros((B,self.ka*C, T), device=x.device)
        padx = F.pad(x,(self.ka//2,self.ka//2))

        for i in range(self.ka):
            out[:, i*C:(i+1)*C, : ] = padx[:, :, i:i+T]
        
        out = out.reshape(B, self.ka ,C , T)
        out = torch.transpose(out, 1,2) 
        out = out.reshape(B, self.ka* C , T)
        
        return out
    
    def init_parameters(self):
        #  shift initialization for group convolution
        kernel = torch.zeros(self.ka, 1, self.ka)
        for i in range(self.ka):
            kernel[i, 0, i] = 1.

        kernel = kernel.repeat(self.dim, 1, 1)
        self.shift_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x, mask):
        B, C, T = x.shape

        _x = self.shift_conv(x)
        
        if self.conv_type == 'gate':
            weight = self.kernel_conv(_x, x, mask)
        else:
            weight, _ = self.kernel_conv(_x, mask) 
            weight = weight.repeat_interleave(self.ka, dim = 1)
        _x = _x*weight
        
        out_conv = self.conv(_x)        
        out_conv = self.norm(out_conv)
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest' )
        else:
            out_mask = mask.to(x.dtype)

        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        
        return out_conv, out_mask
        

if __name__ == "__main__":
    import torch 