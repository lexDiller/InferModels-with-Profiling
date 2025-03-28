from typing import Optional, List, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileOne', 'mobileone', 'reparameterize_model']

class SEBlock(nn.Module):
    """Squeeze and Excite module.
    Pytorch implementation of Squeeze-and-Excitation Networks:
    https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1, stride=1, bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1, stride=1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        x = nn.functional.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = nn.functional.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x

class MobileOneBlock(nn.Module):
    """MobileOne building block.
    Многоветвленная архитектура на этапе тренировки, которая переparameterизуется
    в одноветвленный вариант для инференса.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            self.rbr_skip = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

            rbr_conv = []
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            self.rbr_scale = self._conv_bn(kernel_size=1, padding=0) if kernel_size > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))
        identity_out = self.rbr_skip(x) if self.rbr_skip is not None else 0
        scale_out = self.rbr_scale(x) if self.rbr_scale is not None else 0
        out = scale_out + identity_out
        for branch in self.rbr_conv:
            out += branch(x)
        return self.activation(self.se(out))

    def reparameterize(self):
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')
        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale, bias_scale = (0, 0)
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = nn.functional.pad(kernel_scale, [pad, pad, pad, pad])
        kernel_identity, bias_identity = (0, 0)
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
        kernel_conv, bias_conv = (0, 0)
        for branch in self.rbr_conv:
            _kernel, _bias = self._fuse_bn_tensor(branch)
            kernel_conv += _kernel
            bias_conv += _bias
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # Если branch – просто BatchNorm (skip branch)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            groups=self.groups,
            bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

class MobileOne(nn.Module):
    """MobileOne Model.
    Pytorch implementation of:
    An Improved One millisecond Mobile Backbone – https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                 num_classes: int = 1000,
                 width_multipliers: Optional[List[float]] = None,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        super().__init__()
        assert len(width_multipliers) == 4, "width_multipliers должно содержать 4 значения"
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Stage 0
        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)

    def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int) -> nn.Sequential:
        strides = [2] + [1]*(num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Количество SE блоков не может превышать количество слоёв.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def mobileone(num_classes: int = 1000, inference_mode: bool = False, variant: str = "s0") -> nn.Module:
    PARAMS = {
        "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 4},
        "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
        "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
        "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
        "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},
    }
    variant_params = PARAMS[variant]
    return MobileOne(num_classes=num_classes, inference_mode=inference_mode, **variant_params)

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model
