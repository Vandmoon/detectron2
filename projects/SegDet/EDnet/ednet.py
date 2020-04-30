
import torch.nn.functional as F
from detectron2.layers import (
    Conv2d,
    FrozenBatchNorm2d,
    ShapeSpec,
    get_norm,
)
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from torch import nn

from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7  # added by taolei ,调用detectron2自带的fpn
###lj efficientdet
from .efficientnet import EfficientNet
from .bifpn import BIFPN
from .conv_module import ConvModule
###lj efficientdet

__all__ = [
    "build_EDNet_backbone", "build_retinanet_ednet_fpn_backbone",
]

MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}
FPN_MAP = {
    'efficientdet-d0': {'input_size': 512,
                        'backbone': 'B0',
                        'W_bifpn': 64,
                        'D_bifpn': 2,
                        'D_class': 3},
    'efficientdet-d1': {'input_size': 640,
                        'backbone': 'B1',
                        'W_bifpn': 88,
                        'D_bifpn': 3,
                        'D_class': 3},
    'efficientdet-d2': {'input_size': 768,
                        'backbone': 'B2',
                        'W_bifpn': 112,
                        'D_bifpn': 4,
                        'D_class': 3},
    'efficientdet-d3': {'input_size': 896,
                        'backbone': 'B3',
                        'W_bifpn': 160,
                        'D_bifpn': 5,
                        'D_class': 4},
    'efficientdet-d4': {'input_size': 1024,
                        'backbone': 'B4',
                        'W_bifpn': 224,
                        'D_bifpn': 6,
                        'D_class': 4},
    'efficientdet-d5': {'input_size': 1280,
                        'backbone': 'B5',
                        'W_bifpn': 288,
                        'D_bifpn': 7,
                        'D_class': 4},
    'efficientdet-d6': {'input_size': 1408,
                        'backbone': 'B6',
                        'W_bifpn': 384,
                        'D_bifpn': 8,
                        'D_class': 5},
    'efficientdet-d7': {'input_size': 1636,
                        'backbone': 'B6',
                        'W_bifpn': 384,
                        'D_bifpn': 8,
                        'D_class': 5},
}

class EDNet(Backbone):
    def __init__(self, network='efficientdet-d0', out_features=None):
        super(EDNet, self).__init__()

        p6p7out_channels = 256
        self._size_divisibility = 128  # 128 when using 5 layers of feature maps for BiFPN, while 32 when using only 3 layers
        self._fpn_out_channels = FPN_MAP[network]['W_bifpn']
        self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])
        res_channels = self.backbone.get_list_features()
        #print("******************efficient channels: ", res_channels, '   ', type(res_channels[-1]))
        self._out_feature_strides = {'res3':8, 'res4':16, 'res5':32}
        self._out_feature_channels = {'res3':res_channels[-5], 'res4':res_channels[-3], 'res5':res_channels[-1]}
        self._out_features = out_features

        self.LastLevelP6P7 = LastLevelP6P7(self._out_feature_channels['res5'], p6p7out_channels)
        
        self.neck = BIFPN(in_channels=[self.backbone.get_list_features()[-5], self.backbone.get_list_features()[-3], self.backbone.get_list_features()[-1], p6p7out_channels, p6p7out_channels],
                                out_channels=self._fpn_out_channels,
                                stack=FPN_MAP[network]['D_bifpn'],
                                num_outs=5,
                          norm_cfg = {'type': 'BN'}, 
                          conv_cfg = {'type': 'Conv'},
                          activation = 'relu')

        
        self._fpn_out_features = ['p3', 'p4', 'p5', 'p6', 'p7']
        self._fpn_out_feature_strides = {'p3':8, 'p4':16, 'p5':32, 'p6':64, 'p7':128}
        
        ####################################################
        self._fpn_out_channels = 256  # change into
        self._conv_1x1_256s = nn.ModuleList()
        for i in range(len(self._fpn_out_features)):
            self._conv_1x1_256s.append(ConvModule(
                FPN_MAP[network]['W_bifpn'],
                256,
                1))   
        ####################################################
        
        self._fpn_out_feature_channels = {'p3':self._fpn_out_channels, 'p4':self._fpn_out_channels, 'p5':self._fpn_out_channels, 'p6':self._fpn_out_channels, 'p7':self._fpn_out_channels}
        

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return self._size_divisibility
        
        
    def forward(self, x):
        #print("input size: ", str(x.size()))
        backbone_outputs = []
        y_backbone = self.backbone(x)
        y_p6p7 = self.LastLevelP6P7(y_backbone[-1])

        backbone_outputs.append(y_backbone[-5])
        backbone_outputs.append(y_backbone[-3])
        backbone_outputs.append(y_backbone[-1])
        backbone_outputs.extend(y_p6p7)
        #for b in backbone_outputs:
        #    print("**********", b.size())
        #bifpn
        fpn_output = self.neck(backbone_outputs)
        #for f in fpn_output:
        #    print("&&&&&&&&&&&&&", f.size())
        
        ###############################################################################
        # Add an extra 1x1 conv to change the channel size of each feature map into 256
        fpn_expanded = []
        for i, fpn_feat in enumerate(fpn_output):
            fpn_expanded.append(self._conv_1x1_256s[i](fpn_output[i]))
        ###############################################################################
        
        return dict(zip(self._fpn_out_features, fpn_expanded))  # fpn_output


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._fpn_out_feature_channels[name], stride=self._fpn_out_feature_strides[name]
            )
            for name in self._fpn_out_features
        }

@BACKBONE_REGISTRY.register()
def build_EDNet_backbone(cfg, input_shape):
    print(input_shape.channels)
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    #out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    return EDNet(network='efficientdet-d0', out_features=out_features)



@BACKBONE_REGISTRY.register()
def build_retinanet_ednet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    backbone = build_EDNet_backbone(cfg, input_shape)
    '''
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.out_feature_channels["res5"]
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    '''
    return backbone

