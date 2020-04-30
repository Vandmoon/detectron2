# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
#from . import autoaugdet

from .data_pred import get_keypoint_maps

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True, aug_type = ""):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        ################################################################################################################
        self.autoaugdet = aug_type
        ################################################################################################################
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        
        self.num_classes    = cfg.MODEL.SEG_DET_HEAD.NUM_CLASSES
        
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

#         ################################################################################################################
#         print("AutoAugDet:", dataset_dict["file_name"])
#         h, w, c = image.shape
#         if h <= 0 or w <=0:
#             print("Empty image")
#         if self.autoaugdet and "annotations" in dataset_dict:
#             from detectron2.structures.boxes import BoxMode
#             bboxes = []
#             for label in dataset_dict["annotations"]:
#                 assert label['bbox_mode'] == BoxMode.XYWH_ABS
#                 bboxes.append(label['bbox'])
#             # import cv2, random
#             # showimg_in = image.copy()
#             # for box in bboxes:
#             #     cv2.rectangle(showimg_in, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
#             try:
#                 image, bboxes = autoaugdet.autoaugdet(image, bboxes, self.autoaugdet)
#             except Exception as  e:
#                 print("AutoAug Error:", e)
#             # showimg_out = image.copy()
#             # for box in bboxes:
#             #     cv2.rectangle(showimg_out, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
#             # cv2.imshow("in", showimg_in)
#             # cv2.imshow("out", showimg_out)
#             # cv2.waitKey(0)
#             for i in range(len(bboxes)):
#                 dataset_dict["annotations"][i]['bbox'] = bboxes[i]

#         #################################################################################################       
        
        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)

            image, transforms = T.apply_transform_gens(self.tfm_gens, image)

            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            
            # generate heatmaps of keypoints
            #if dataset_dict["instances"].has("gt_keypoints"):
            #
            
            #For segmentation-based detection, transform the instance-level segmentation mask into semantic segmasks and contour maps
            # turning instance-level segmentation map into semantic segmap            
            # get the contour map for segmentation-based detection            
            dataset_dict["contours"], dataset_dict["semseg"] = utils.annotations_to_segmaps(annos, self.num_classes, image_shape)
            
            kpts = [obj.get("keypoints", []) for obj in annos]
            map_shape = (image_shape[0], image_shape[1])
            kp_maps, short_offsets = get_keypoint_maps(None, kpts, map_shape)
            dataset_dict["kp_maps"] = kp_maps.transpose(2, 0, 1)
            dataset_dict["short_offsets"] = short_offsets.transpose(2, 0, 1)
            
            ################################################################
#             # visualize the keypoints
#             from detectron2.utils.visualizer import Visualizer
#             from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
#             from os import path

#             image_rgb = image[..., ::-1]
#             V = Visualizer(image_rgb, dataset_dict)
#             # draw the foreground mask of each object category
#             binary_masks = kp_maps>0.1
#             _, fn = path.split(dataset_dict["file_name"])
#             fn_next, ext = path.splitext(fn)
#             print('Mask size: ', binary_masks.shape)
#             print('Image size: ', image_rgb.shape)
#             assert binary_masks.shape[1]==image_rgb.shape[0], (binary_masks.shape[1], image_rgb.shape[0])
#             assert binary_masks.shape[2]==image_rgb.shape[1], (binary_masks.shape[2], image_rgb.shape[1])
#             assert image_rgb.shape[2]==3, image_rgb.shape[2]
#             bm = binary_masks

#             for i in range(binary_masks.shape[0]):
#                 masked_image = V.draw_binary_mask(
#                     bm[i, :, :].squeeze(), color=None, edge_color='r', alpha=0.5, area_threshold=10
#                 ) # COCO_CATEGORIES[i]["color"]
# #                 filepath = "tmp/" + fn_next + '_' + COCO_CATEGORIES[i]["name"] + '.png'
# #                 masked_image.save(filepath)
#             filepath = "tmp/" + fn_next + '.png'
#             masked_image.save(filepath)
            ################################################################
            
            ################################################
#             # visualize the segmentation mask
#             from os import path
#             image_rgb = image[..., ::-1]  #utils.read_image(dataset_dict["file_name"], format="RGB")
#             segmask = dataset_dict["semseg"].tensor.numpy()
#             _, fn = path.split(dataset_dict["file_name"])
#             fn_next, ext = path.splitext(fn)
            
#             im = Image.fromarray(np.uint8(image_rgb))
#             filepath = "tmp_segmap_sorted/" + fn_next + '_raw.png'
#             im.save(filepath)
            
#             im2 = Image.fromarray(np.uint8(segmask*3))
#             filepath2 = "tmp_segmap_sorted/" + fn_next + '_seg.png'
#             im2.save(filepath2)
            
            ################################################
            
            ###############
#             # visualize the segmentation map and contours
#             from detectron2.utils.visualizer import Visualizer
#             from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
#             from os import path
#             #V.draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8)            
#             image_rgb = image[..., ::-1]  #utils.read_image(dataset_dict["file_name"], format="RGB")
#             V = Visualizer(image_rgb, dataset_dict)
#             # draw the foreground mask of each object category
#             #binary_masks = dataset_dict["contours"].gt_segmasks.tensor
#             binary_masks = dataset_dict["contours"].gt_contours.tensor
#             _, fn = path.split(dataset_dict["file_name"])
#             fn_next, ext = path.splitext(fn)
#             print('Mask size: ', binary_masks.size())
#             print('Image size: ', image_rgb.shape)
#             assert binary_masks.size(1)==image_rgb.shape[0], (binary_masks.size(1), image_rgb.shape[0])
#             assert binary_masks.size(2)==image_rgb.shape[1], (binary_masks.size(2), image_rgb.shape[1])
#             assert image_rgb.shape[2]==3, image_rgb.shape[2]
#             bm = binary_masks.numpy()
# #             bm_uint8 = bm.astype("uint8")
# #             print(bm)
#             for i in range(binary_masks.size(0)):
#                 masked_image = V.draw_binary_mask(
#                     bm[i, :, :].squeeze(), color=None, edge_color='r', alpha=0.5, area_threshold=10
#                 ) # COCO_CATEGORIES[i]["color"]
# #                 filepath = "tmp/" + fn_next + '_' + COCO_CATEGORIES[i]["name"] + '.png'
# #                 masked_image.save(filepath)
#             filepath = "tmp/" + fn_next + '.png'
#             masked_image.save(filepath)
            
################################################################################################# 

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
            
        return dataset_dict
