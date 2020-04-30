# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone import build_backbone
from .postprocessing import sem_seg_postprocess
from detectron2.modeling import META_ARCH_REGISTRY

__all__ = ["SegDetector", "SEG_DET_HEADS_REGISTRY", "SegDetFPNHead", "build_seg_det_head"]


SEG_DET_HEADS_REGISTRY = Registry("SEG_DET_HEADS")
"""
Registry for segmentation-based detection heads, which make segmentation and detection predictions
from feature maps.
"""


@META_ARCH_REGISTRY.register()
class SegDetector(nn.Module):
    """
    Main class for segmentation-based detector architectures.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_seg_det_head(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg: semantic segmentation ground truth
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "sem_seg" whose value is a
                Tensor of the output resolution that represents the
                per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

#         if "sem_seg" in batched_inputs[0]:
#             targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
#             targets = ImageList.from_tensors(
#                 targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
#             ).tensor
#         else:
#             targets = None
            
        if "contours" in batched_inputs[0]:
#             for i, x in enumerate(batched_inputs):
#                 print("contour size {}: {}".format(i, x["contours"].gt_contours.to(self.device).tensor.size()))
#                 print("segmask size {}: {}".format(i, x["contours"].gt_segmasks.to(self.device).tensor.size()))
            #contours = torch.cat([x["contours"].gt_contours.to(self.device).tensor for x in batched_inputs])
            #segmasks = torch.cat([x["contours"].gt_segmasks.to(self.device).tensor for x in batched_inputs])
            contours = ImageList.from_tensors(
                [x["contours"].gt_contours.to(self.device).tensor for x in batched_inputs], self.backbone.size_divisibility
            ).tensor
            segmasks = ImageList.from_tensors(
                [x["contours"].gt_segmasks.to(self.device).tensor for x in batched_inputs], self.backbone.size_divisibility
            ).tensor
#             semseg = ImageList.from_tensors(
#                 [x["semseg"].to(self.device).tensor for x in batched_inputs], self.backbone.size_divisibility, self.sem_seg_head.ignore_value
#             ).tensor
        else:
            contours = None
            segmasks = None
        
        if "instances" in batched_inputs[0]:
            objmask = [x["instances"].gt_masks.to(self.device).tensor for x in batched_inputs]
#             for i, x in enumerate(batched_inputs):
#                 print("gt masks size {}: {}".format(i, x["instances"].gt_masks.tensor.size()))
#                 print("gt classes size {}: {}".format(i, x["instances"].gt_classes.size()))
#             objmask = ImageList.from_tensors(
#                 [x["instances"].gt_masks.to(self.device).tensor for x in batched_inputs], self.backbone.size_divisibility, self.sem_seg_head.ignore_value
#             ).tensor
#             print("objmask size: {}".format(len(objmask)))
            classes = [x["instances"].gt_classes.to(self.device) for x in batched_inputs]
        else:
            objmask = None
            classes = None         
        
        results, losses = self.sem_seg_head(features, segmasks, contours, objmask, classes)        

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            
            #TODO: translate semantic segmentations and contour maps into detection bounding boxes
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results


def build_seg_det_head(cfg, input_shape):
    """
    Build a seg-det head from `cfg.MODEL.SEG_DET_HEAD.NAME`.
    """
    name = cfg.MODEL.SEG_DET_HEAD.NAME
    return SEG_DET_HEADS_REGISTRY.get(name)(cfg, input_shape)


def embedding_loss(pred_emb, gt_objmask, gt_classes, w_intra=1.0, w_inter=1.0, w_reg=0.1):
    """
    Calculate the embedding loss from a batch of class-specific embedding predictions and groundtruth instance-specific segmentation.
    Args:
        pred_emb: Tensor, embbeding prediction in (N, C, H, W) format, 
            where N is the batch size, C is the number of object classes, (H, W) is the size of output.
        gt_objmask: Tensor, groundtruth instance-specific segmentation mask in (N, K, H, W) format, 
            where N is the batch size, K is the number of instances, (H, W) is the size of output.
        gt_classes: Tensor, groundtruth classes of instances in (N, K) format, 
            where N is the batch size, K is the number of instances.
    Output:
        loss: Tensor, embedding loss of 1 element.
    """
    N, C, H, W = pred_emb.size()
    if not N:
        return 0
    
    assert(N == len(gt_objmask))
    assert(N == len(gt_classes))
    
    # padding gt_objmask to the disired image size (H, W)
    padded_objmask = []
    for m in gt_objmask:
        padded_objmask.append(F.pad(m, (0, W - m.size(-1), 0, H - m.size(-2))))
    
    loss_intra_var = 0
    loss_inter_diff = 0
    loss_reg = 0
    for i in range(N):
        K = len(gt_classes[i])
        if not K:
            continue
        obj_means = np.zeros(K, dtype=np.float)
        for j in range(K):
            emb = pred_emb[i,gt_classes[i][j]]
            emb_masked = torch.masked_select(emb, padded_objmask[i][j])
            obj_means[j] = emb_masked.mean()
            obj_var = emb_masked.var()
            loss_intra_var += obj_var/K
        
        loss_inter_diff_i = 0
        npair = 0
        for j in range(K-1):
            for k in range(j+1, K):
                if torch.eq(gt_classes[i][j], gt_classes[i][k]):
                    loss_inter_diff_i += max(1 - np.power(obj_means[j]-obj_means[k], 2), 0)
                    npair += 1
        if npair:
            loss_inter_diff += loss_inter_diff_i / npair
        
        # add a regularizer
        loss_reg = np.power(obj_means, 2).mean()
    
    loss = w_reg*loss_reg/N  # w_inter*loss_inter_diff/N  #w_intra*loss_intra_var/N  + + 
    return loss
    

@SEG_DET_HEADS_REGISTRY.register()
class SegDetFPNHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEG_DET_HEAD.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value     = cfg.MODEL.SEG_DET_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEG_DET_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEG_DET_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEG_DET_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.SEG_DET_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEG_DET_HEAD.LOSS_WEIGHT
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor_segmap = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        self.predictor_contour = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Embedding to separate different objects of the same category
        self.predictor_emb = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        
        weight_init.c2_msra_fill(self.predictor_segmap)
        weight_init.c2_msra_fill(self.predictor_contour)
        weight_init.c2_msra_fill(self.predictor_emb)

    def forward(self, features, gt_segmap=None, gt_contour=None, gt_objmask=None, gt_classes=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        segmap = self.predictor_segmap(x)
        segmap = F.interpolate(segmap, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        
        contour = self.predictor_contour(x)
        contour = F.interpolate(contour, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        
        # Embedding
        emb = self.predictor_emb(x)
        emb = F.interpolate(emb, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        
        if self.training:
            losses = {}
                      
#             losses["loss_seg_map"] = (
#                 F.binary_cross_entropy_with_logits(segmap, gt_segmap, reduction="mean")
#                 * self.loss_weight
#             )
            
            
#             losses["loss_contour"] = (
#                 F.binary_cross_entropy_with_logits(contour, gt_contour, reduction="mean")
#                 * self.loss_weight
#             )

            
            # Embedding loss, including intra-class variance, inter-class distance, and regularization loss
            losses["embedding"] = (
                embedding_loss(emb, gt_objmask, gt_classes)
                * self.loss_weight
            )
            
            #TODO: consider to add a loss that measures the inconsistency between the embedding and the segmap or contour map
            #print("losses: ".format(losses))
            
            return [], losses
        else:
            #TODO: combine segmap and contour into detections
            return [segmap, contour, emb], {}
