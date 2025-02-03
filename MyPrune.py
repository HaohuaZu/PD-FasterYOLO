import torch
import torch_pruning as tp
from typing import Sequence

from models.yolo import RepConv
class RepConvPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: RepConv, idxs: Sequence[int]):
        # print(f'RepConvPruner prune_out_channels:{idxs}')
        # prune for rbr_dense
        tp.prune_conv_out_channels(layer.rbr_dense[0], idxs)
        tp.prune_batchnorm_out_channels(layer.rbr_dense[1], idxs)
        
        # prune for rbr_1x1
        tp.prune_conv_out_channels(layer.rbr_1x1[0], idxs)
        tp.prune_batchnorm_out_channels(layer.rbr_1x1[1], idxs)
        
        layer.out_channels = layer.out_channels - len(idxs)
        return layer
            
    def prune_in_channels(self, layer: RepConv, idxs: Sequence[int]):
        # print(f'RepConvPruner prune_in_channels:{idxs}')
        # prune for rbr_dense
        tp.prune_conv_in_channels(layer.rbr_dense[0], idxs)
        
        # prune for rbr_1x1
        tp.prune_conv_in_channels(layer.rbr_1x1[0], idxs)
        
        # prune for rbr_identity
        if layer.rbr_identity is not None: tp.prune_batchnorm_in_channels(layer.rbr_identity, idxs)
        
        layer.in_channels = layer.in_channels - len(idxs)
        return layer
        
    def get_out_channels(self, layer: RepConv):
        return layer.out_channels
    
    def get_in_channels(self, layer: RepConv):
        return layer.in_channels
    
    def get_channel_groups(self, layer: RepConv):
        return layer.groups

from models.convnextv2 import LayerNorm
class LayerNormPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer:LayerNorm, idxs: Sequence[int]):
        num_features = layer.normalized_shape[0]
        keep_idxs = torch.tensor(list(set(range(num_features)) - set(idxs)))
        keep_idxs.sort()
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, -1)
        layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, -1)
        layer.normalized_shape = (len(keep_idxs),)
    
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer):
        return layer.normalized_shape[0]

    def get_in_channels(self, layer):
        return layer.normalized_shape[0]

from models.dyhead_prune import DyHeadBlock_Prune
class DyHeadBlockPruner(tp.pruner.BasePruningFunc):
    def prune_in_channels(self, layer: DyHeadBlock_Prune, idxs: Sequence[int]):
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        
        layer.spatial_conv_low.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_low.conv.weight, keep_idxs, 1)
        layer.spatial_conv_mid.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_mid.conv.weight, keep_idxs, 1)
        layer.spatial_conv_high.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_high.conv.weight, keep_idxs, 1)
        tp.prune_conv_in_channels(layer.spatial_conv_offset, idxs)
        
        return layer
    
    def prune_out_channels(self, layer: DyHeadBlock_Prune, idxs: Sequence[int]):
        keep_idxs = list(set(range(layer.spatial_conv_low.conv.weight.size(0))) - set(idxs))
        keep_idxs.sort()
        keep_idxs = keep_idxs[:len(keep_idxs) - (len(keep_idxs) % self.get_out_channel_groups(layer))]
        idxs = list(set(range(layer.spatial_conv_low.conv.weight.size(0))) - set(keep_idxs))
        
        # spatial_conv
        layer.spatial_conv_low.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_low.conv.weight, keep_idxs, 0)
        layer.spatial_conv_mid.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_mid.conv.weight, keep_idxs, 0)
        layer.spatial_conv_high.conv.weight = self._prune_parameter_and_grad(layer.spatial_conv_high.conv.weight, keep_idxs, 0)
        layer.spatial_conv_low.norm = self.prune_groupnorm(layer.spatial_conv_low.norm, keep_idxs)
        layer.spatial_conv_mid.norm = self.prune_groupnorm(layer.spatial_conv_mid.norm, keep_idxs)
        layer.spatial_conv_high.norm = self.prune_groupnorm(layer.spatial_conv_high.norm, keep_idxs)
        
        # scale_attn_module
        tp.prune_conv_in_channels(layer.scale_attn_module[1], idxs)
        
        # task_attn_module
        dim = layer.task_attn_module.oup
        idxs_repeated = idxs + \
            [i+dim for i in idxs] + \
            [i+2*dim for i in idxs] + \
            [i+3*dim for i in idxs]
        tp.prune_linear_in_channels(layer.task_attn_module.fc[0], idxs)
        tp.prune_linear_out_channels(layer.task_attn_module.fc[2], idxs_repeated)
        layer.task_attn_module.oup = layer.task_attn_module.oup - len(idxs)
        
        return layer
    
    def get_out_channels(self, layer: DyHeadBlock_Prune):
        return layer.spatial_conv_low.conv.weight.size(0)

    def get_in_channels(self, layer: DyHeadBlock_Prune):
        return layer.spatial_conv_low.conv.weight.size(1)
    
    def get_in_channel_groups(self, layer: DyHeadBlock_Prune):
        return 1
    
    def get_out_channel_groups(self, layer: DyHeadBlock_Prune):
        return layer.spatial_conv_high.norm.num_groups
    
    def prune_groupnorm(self, layer: torch.nn.GroupNorm, keep_idxs):
        layer.num_channels = keep_idxs
        if layer.affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer