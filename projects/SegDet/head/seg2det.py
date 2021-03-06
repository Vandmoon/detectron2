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
from projects.SegDet.postprocess.postprocessing import seg_det_postprocess
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
           
        if "contours" in batched_inputs[0]:
            contours = ImageList.from_tensors(
                [x["contours"].gt_contours.to(self.device).tensor for x in batched_inputs], self.backbone.size_divisibility
            ).tensor
#             segmasks = ImageList.from_tensors(
#                 [x["contours"].gt_segmasks.to(self.device).tensor for x in batched_inputs], self.backbone.size_divisibility
#             ).tensor
            segmasks = ImageList.from_tensors(
                [x["semseg"].to(self.device).tensor.long() for x in batched_inputs], self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
    
            kp_maps = ImageList.from_tensors(
                [torch.tensor(x["kp_maps"]).to(self.device) for x in batched_inputs], self.backbone.size_divisibility
            ).tensor
            
            short_offsets = ImageList.from_tensors(
                [torch.tensor(x["short_offsets"]).to(self.device).float() for x in batched_inputs], self.backbone.size_divisibility
            ).tensor
        else:
            contours = None
            segmasks = None
            
            kp_maps = None
            short_offsets = None
        
        if "instances" in batched_inputs[0]:
            objmask = [x["instances"].gt_masks.to(self.device).tensor for x in batched_inputs]
            classes = [x["instances"].gt_classes.to(self.device) for x in batched_inputs]
        else:
            objmask = None
            classes = None         

        results, losses = self.sem_seg_head(features, segmasks, contours, objmask, classes, kp_maps, short_offsets) 

        if self.training:
            return losses

        processed_results = []
        for segmap, contour, emb, kp_map, short_offsets, input_per_image, image_size in zip(results[0], results[1], results[2], results[3], results[4], batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            
            #TODO: translate semantic segmentations and contour maps into detection bounding boxes
            r = seg_det_postprocess(segmap, contour, emb, image_size, height, width)
            processed_results.append({"instances": r, "segmap": segmap, "contour": contour, "emb": emb, "kp_map": kp_map, "short_offsets": short_offsets, "input": input_per_image})  #
        return processed_results


def build_seg_det_head(cfg, input_shape):
    """
    Build a seg-det head from `cfg.MODEL.SEG_DET_HEAD.NAME`.
    """
    name = cfg.MODEL.SEG_DET_HEAD.NAME
    return SEG_DET_HEADS_REGISTRY.get(name)(cfg, input_shape)


class Embedding_loss(nn.Module):
    def __init__(self, cfg, w_intra=1.0, w_inter=1.0, w_reg=1):
        super().__init__()
        
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.w_intra = w_intra
        self.w_inter = w_inter
        self.w_reg = w_reg
        
        self.to(self.device)
        
    def forward(self, pred_emb, gt_objmask, gt_classes):
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
            loss: Tensor, embedding loss of 1 element."""
        pred_emb.to(self.device)
        gt_objmask.to(self.device)
        gt_classes.to(self.device)
        
        C, H, W = pred_emb.size()
        K = len(gt_objmask)
        if not K:
            return torch.zeros(1).to(self.device)
        assert(K == len(gt_classes))
#         print("K: ", K)

        # padding gt_objmask to the disired image size (H, W)
        padded_objmask = F.pad(gt_objmask, (0, W - gt_objmask.size(-1), 0, H - gt_objmask.size(-2)))

        loss_intra_var = torch.zeros(K).to(self.device)
        loss_inter_diff = torch.zeros(1).to(self.device)

        obj_means = torch.zeros(K).to(self.device)
        for j in range(K):
            emb = pred_emb[gt_classes[j]]
            emb_masked = torch.masked_select(emb, padded_objmask[j])
            if len(emb_masked):
                obj_means[j] = emb_masked.mean()
                loss_intra_var[j] = emb_masked.var(unbiased=False)

        for j in range(K-1):
            for k in range(j+1, K):
                if torch.eq(gt_classes[j], gt_classes[k]):
                    loss_inter_diff += max(1 - torch.pow(obj_means[j]-obj_means[k], 2), 0)

        # add a regularizer
        loss_reg = torch.pow(obj_means, 2).mean()

        loss = self.w_inter*loss_inter_diff + self.w_reg*loss_reg +self.w_intra*loss_intra_var.mean() 
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
        
        self.num_keypoints     = 17
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
        
        # NOTE: added a background class
        self.predictor_segmap = Conv2d(conv_dims, num_classes+1, kernel_size=1, stride=1, padding=0)
        self.predictor_contour = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Embedding to separate different objects of the same category
        self.predictor_emb = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        
        # keypoint heatmap
        self.predictor_kp_maps = Conv2d(conv_dims, self.num_keypoints, kernel_size=1, stride=1, padding=0)
        
        # short-range offsets to keypoints
        self.predictor_short_offsets = Conv2d(conv_dims, self.num_keypoints*2, kernel_size=1, stride=1, padding=0)
        
        weight_init.c2_msra_fill(self.predictor_segmap)
        weight_init.c2_msra_fill(self.predictor_contour)
        weight_init.c2_msra_fill(self.predictor_emb)
        
        weight_init.c2_msra_fill(self.predictor_kp_maps)
        weight_init.c2_msra_fill(self.predictor_short_offsets)
        
        self.emb_loss = Embedding_loss(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        self.num_classes = num_classes
        

    def forward(self, features, gt_segmap=None, gt_contour=None, gt_objmask=None, gt_classes=None, gt_kp_maps=None, gt_short_offsets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        segmap = self.predictor_segmap(x)
        segmap = F.interpolate(segmap, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
#         # TODO: add a pointwise softmax here for inter-class competition
#         segmap = F.softmax(segmap, dim=-3)
        
        contour = self.predictor_contour(x)
        contour = F.interpolate(contour, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        #contour = F.softmax(contour, dim=-3)  # the contours are not mutual exclusive between object classes
        
        # Embedding
        emb = self.predictor_emb(x)
        emb = F.interpolate(emb, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        
        # keypoint heatmap
        kp_maps = self.predictor_kp_maps(x)
#         kp_maps = F.sigmoid(kp_maps)
        kp_maps = F.interpolate(kp_maps, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        
        # short-range offsets to keypoints
        short_offsets = self.predictor_short_offsets(x)
        short_offsets = F.interpolate(short_offsets, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        
        if self.training:
            losses = {}
            
            #pos_weight_s = torch.ones([self.num_classes+1])  # *10
            pos_weight_s = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 10, 10, 3, 10, 10, 10, 10, 3, 10, 10, 10, 1, 3, 3, 3, 1, 10, 10, 10, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 10, 10, 1, 1, 3, 1, 10, 1, 1, 10, 1, 1, 10, 1, 10, 10, 0.3])
#             pos_weight_s = torch.reshape(pos_weight_s,(1,self.num_classes,1,1))
#             assert segmap.size(1)==self.num_classes, segmap.size()
#             losses["loss_seg_map"] = (
#                 F.binary_cross_entropy_with_logits(segmap, gt_segmap, reduction="mean", pos_weight=pos_weight_s.to(self.device))
#                 * self.loss_weight
#             )
            # after softmax the segmap are already probabilities
            # TODO: add weights to bias towards small objects such as spoons
#             losses["loss_seg_map"] = (
#                 F.binary_cross_entropy(segmap, gt_segmap, reduction="mean")
#                 * self.loss_weight
#             )
            losses["loss_sem_seg"] = (
                F.cross_entropy(segmap, gt_segmap, reduction="mean", ignore_index=self.ignore_value)
                * self.loss_weight
            )  # weight=pos_weight_s.to(self.device), 
            #print("loss_seg_map: ", losses["loss_seg_map"])
            
            pos_weight_c = torch.ones([self.num_classes])*20  # 20
            pos_weight_c = torch.reshape(pos_weight_c,(1,self.num_classes,1,1))
            losses["loss_contour"] = (
                F.binary_cross_entropy_with_logits(contour, gt_contour, reduction="mean", pos_weight=pos_weight_c.to(self.device))
                * self.loss_weight * 10
            )
            
            # Embedding loss, including intra-class variance, inter-class distance, and regularization loss
            loss_emb1d = torch.zeros(emb.size(0)).to(self.device)
            for i in range(emb.size(0)):
                loss_emb1d[i] = self.emb_loss(emb[i].squeeze(dim=0), gt_objmask[i], gt_classes[i])  # /emb.size(0)
            
            if len(loss_emb1d):
                losses["embedding"] = loss_emb1d.mean() * self.loss_weight * 0.1
            else:
                losses["embedding"] = torch.tensor(0.).to(self.device)
            
            # losses of keypoint heatmap
            losses["loss_kp_maps"] = (
                F.binary_cross_entropy_with_logits(kp_maps, gt_kp_maps, reduction="mean")
                * self.loss_weight
            )
                
            # losses of short offsets to keypoints
            # TODO: we may ONLY account for the losses inside keypoint discs as suggested by PersonLab
            losses["loss_short_offsets"] = (
                F.l1_loss(short_offsets, gt_short_offsets, reduction="mean")
                * self.loss_weight
            )
                
            return [], losses
        else:
            #TODO: combine segmap and contour into detections
            # turning segmap logits into probabilities
            segmap = F.softmax(segmap, dim=-3)  # F.sigmoid(segmap)  # use sigmoid on logits ONLY
            contour = F.sigmoid(contour)
            kp_maps = F.sigmoid(kp_maps)
            
            return [segmap, contour, emb, kp_maps, short_offsets], {}
