import timm
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONES


@BACKBONES.register_module()
class UNetFormerBackbone(BaseModule):
    """UNetFormer encoder backbone using timm models.

    Wraps a timm feature extractor to produce multi-scale feature maps
    compatible with the MMSegmentation pipeline.

    Args:
        backbone_name (str): Name of the timm model.
            Default: 'swsl_resnet18'.
        pretrained (bool): Whether to use pretrained weights.
            Default: True.
        out_indices (tuple[int]): Indices of output feature levels from timm
            (0-indexed; 0=stem, 1=layer1, ..., 4=layer4).
            Default: (1, 2, 3, 4) to skip the stem and output layers 1-4.
        output_stride (int): Output stride of the backbone. Only passed
            to timm when != 32 to avoid unnecessary dilation injection.
            Default: 32.
        norm_eval (bool): Whether to set BatchNorm layers to eval mode
            during training to preserve pretrained statistics.
            Default: False.
    """

    def __init__(self,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 out_indices=(1, 2, 3, 4),
                 output_stride=32,
                 norm_eval=False,
                 init_cfg=None):
        super(UNetFormerBackbone, self).__init__(init_cfg)
        self.norm_eval = norm_eval

        kwargs = dict(
            features_only=True,
            out_indices=out_indices,
            pretrained=pretrained)
        # Only pass output_stride when non-default to avoid timm switching
        # to hook-based feature extraction unnecessarily
        if output_stride != 32:
            kwargs['output_stride'] = output_stride

        self.encoder = timm.create_model(backbone_name, **kwargs)
        self.out_channels = self.encoder.feature_info.channels()

    def forward(self, x):
        features = self.encoder(x)
        return tuple(features)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
