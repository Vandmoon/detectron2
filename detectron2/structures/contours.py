# Create contour mask.
import copy
import itertools
import numpy as np
from typing import Any, Iterator, List, Union
import pycocotools.mask as mask_utils
import torch

from detectron2.layers import cat
from detectron2.layers.roi_align import ROIAlign

from .masks import PolygonMasks, polygons_to_bitmask

import cv2


def polygons_to_contours(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of contour (height, width)
    """
    contours = np.zeros([height, width], np.uint8)
    for p in polygons:
        #print("p: ", p)
        cv2.polylines(contours, np.int32(p.reshape(1, -1, 2)), 1, 255)
        #print("eop")
    
    return contours


def contour_distance_transform(contours: np.ndarray, thr_dist = 5) -> np.ndarray:
    """
    Args:
        contours: a mask of contours

    Returns:
        ndarray: distance map to contours
    """
    contours = 255-contours.astype(np.uint8)
    dist_map = cv2.distanceTransform(contours, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # truncate too large distances
    dist_map = np.minimum(dist_map, thr_dist)
    
    # TODO: normalize to [0, 1]
    dist_map = 1 - dist_map / thr_dist  #dist_map.max()  # float32 ndarray
    
    #ret, sure_fg = cv2.threshold(dist_map,keep_ratio*dist_map.max(),255,0)
    return dist_map


class ContourMaps:
    """
    This class stores the contour maps for all object classes in one image, in the form of distance-to-boader maps.

    Attributes:
        tensor_contour: bool Tensor of C,H,W representing contour maps of C object categories in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: [0, 1] float Tensor of C,H,W representing contour maps of C object categories in the image.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float, device=device)
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, device: str) -> "ContourMaps":
        return ContourMaps(self.tensor.to(device))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "ContourMaps":
        """
        Returns:
            ContourMaps: Create a new :class:`ContourMaps` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `ContourMaps` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return ContourMaps(self.tensor[item].view(1, -1)) 
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on ContourMaps with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return ContourMaps(m)

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    @staticmethod
    def from_polygons(
        polygons: List[List[np.ndarray]], category_ids: List[int], height: int, width: int, num_classes: int, thr_dist = 3
    ) -> "ContourMaps":
        """
        Args:
            polygons (list[list[ndarray]]): polygons of instances in an image.
            category_ids: List[int]: category ids of instances in an image.
            height, width (int): height and width of the image.
            num_classes (int): all possible object classes in the whole dataset.
            thr_dist: maximum distance in the output distance map.
        """
        assert(len(polygons)==len(category_ids))
        contours = np.zeros([num_classes, height, width], np.float)
        for i, p in enumerate(polygons):
            cls = category_ids[i]
            assert(cls >=0 and cls < num_classes)
#             bitmask = polygons_to_bitmask(obj["segmentation"], height, width)
#             sem_seg[:, :, cls] = np.logical_or(sem_seg[:, :, cls], bitmask)
            
            contour = polygons_to_contours(p, height, width)
            contour_map = contour_distance_transform(contour, thr_dist = thr_dist)
            contours[cls, :, :] = np.maximum(contours[cls, :, :], contour_map)
        
        return ContourMaps(torch.from_numpy(contours))

#     def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
#         """
#         Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
#         This can be used to prepare training targets for Mask R-CNN.
#         It has less reconstruction error compared to rasterization with polygons.
#         However we observe no difference in accuracy,
#         but ContourMaps requires more memory to store all the masks.

#         Args:
#             boxes (Tensor): Nx4 tensor storing the boxes for each mask
#             mask_size (int): the size of the rasterized mask.

#         Returns:
#             Tensor:
#                 A bool tensor of shape (N, mask_size, mask_size), where
#                 N is the number of predicted boxes for this image.
#         """
#         assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
#         device = self.tensor.device

#         batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
#         rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

#         bit_masks = self.tensor.to(dtype=torch.float32)
#         rois = rois.to(device=device)
#         output = (
#             ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
#             .forward(bit_masks[:, None, :, :], rois)
#             .squeeze(1)
#         )
#         output = output >= 0.5
#         return output

    def get_bounding_boxes(self) -> None:
        # not needed now
        raise NotImplementedError

    @staticmethod
    def cat(contourmaps_list: List["ContourMaps"]) -> "ContourMaps":
        """
        Concatenates a list of ContourMaps into a single ContourMaps

        Arguments:
            contourmaps_list (list[ContourMaps])

        Returns:
            ContourMaps: the concatenated ContourMaps
        """
        assert isinstance(contourmaps_list, (list, tuple))
        assert len(contourmaps_list) > 0
        assert all(isinstance(contourmaps, ContourMaps) for contourmaps in contourmaps_list)

        cat_contourmaps = type(contourmaps_list[0])(cat([cm.tensor for cm in contourmaps_list], dim=0))
        return cat_contourmaps


class SegMasks:
    """
    This class stores the semantic segmentation for all object classes in one image.

    Attributes:
        tensor_sem_seg: bool Tensor of H,W,C representing semantic segmentations of C object categories in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: [0, 1] float Tensor of H,W,C, representing contour maps of C object categories in the image.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float, device=device)
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, device: str) -> "SegMasks":
        return SegMasks(self.tensor.to(device))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "SegMasks":
        """
        Returns:
            SegMasks: Create a new :class:`SegMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `ContourMaps` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return SegMasks(self.tensor[item].view(1, -1))
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on SegMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return SegMasks(m)

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    @staticmethod
    def from_polygons(
        polygons: List[List[np.ndarray]], category_ids: List[int], height: int, width: int, num_classes: int) -> "SegMasks":
        """
        Args:
            polygons (list[list[ndarray]]): polygons of instances in an image.
            category_ids: List[int]: category ids of instances in an image.
            height, width (int): height and width of the image.
            num_classes (int): all possible object classes in the whole dataset.
        """
        assert(len(polygons)==len(category_ids))
        sem_seg = np.zeros([num_classes, height, width], np.bool)
        for i, p in enumerate(polygons):
            cls = category_ids[i]
            assert(cls >=0 and cls < num_classes)
            bitmask = polygons_to_bitmask(p, height, width)
            sem_seg[cls, :, :] = np.logical_or(sem_seg[cls, :, :], bitmask)
        
        return SegMasks(torch.from_numpy(sem_seg))

#     def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
#         """
#         Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
#         This can be used to prepare training targets for Mask R-CNN.
#         It has less reconstruction error compared to rasterization with polygons.
#         However we observe no difference in accuracy,
#         but ContourMaps requires more memory to store all the masks.

#         Args:
#             boxes (Tensor): Nx4 tensor storing the boxes for each mask
#             mask_size (int): the size of the rasterized mask.

#         Returns:
#             Tensor:
#                 A bool tensor of shape (N, mask_size, mask_size), where
#                 N is the number of predicted boxes for this image.
#         """
#         assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
#         device = self.tensor.device

#         batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
#         rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

#         bit_masks = self.tensor.to(dtype=torch.float32)
#         rois = rois.to(device=device)
#         output = (
#             ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
#             .forward(bit_masks[:, None, :, :], rois)
#             .squeeze(1)
#         )
#         output = output >= 0.5
#         return output

    def get_bounding_boxes(self) -> None:
        # not needed now
        raise NotImplementedError

    @staticmethod
    def cat(segmasks_list: List["SegMasks"]) -> "SegMasks":
        """
        Concatenates a list of SegMasks into a single SegMasks

        Arguments:
            segmasks_list (list[SegMasks])

        Returns:
            SegMasks: the concatenated SegMasks
        """
        assert isinstance(segmasks_list, (list, tuple))
        assert len(segmasks_list) > 0
        assert all(isinstance(segmasks, SegMasks) for segmasks in segmasks_list)

        cat_segmasks = type(segmasks_list[0])(cat([cm.tensor for cm in segmasks_list], dim=0))
        return cat_segmasks
    
    
class SemSegMask:
    """
    This class stores the semantic segmentation for all object classes in one image.

    Attributes:
        tensor_sem_seg: bool Tensor of H,W representing semantic segmentations of all object categories in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: [0, ncls-1] uint8 Tensor of H,W, representing contour maps of all object categories in the image.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float, device=device)
        assert tensor.dim() == 2, tensor.size()
        self.image_size = tensor.shape
        self.tensor = tensor

    def to(self, device: str) -> "SemSegMask":
        return SemSegMask(self.tensor.to(device))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        s = self.__class__.__name__
        return s

    def __len__(self) -> int:
        return 1
    
    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.any()

    @staticmethod
    def from_polygons(
        polygons: List[List[np.ndarray]], category_ids: List[int], height: int, width: int, num_classes: int) -> "SemSegMask":
        """
        Args:
            polygons (list[list[ndarray]]): polygons of instances in an image.
            category_ids: List[int]: category ids of instances in an image.
            height, width (int): height and width of the image.
            num_classes (int): all possible object classes in the whole dataset.
        """
        assert(len(polygons)==len(category_ids))
        sem_seg = np.ones([height, width], np.uint8)*num_classes  # set the background to C instead of 255
        
        #NOTE: sort the objects by size, paint the large object first to avoid covering the small ones
        bitmasks = []
        npixels = []
        for i, p in enumerate(polygons):
            bitmask = polygons_to_bitmask(p, height, width)
            bitmasks.append(bitmask)
            npixels.append(np.sum(bitmask.astype(np.float)))
        
        isort = sorted(range(len(npixels)), key=lambda k: npixels[k], reverse=True)
        for i in range(len(npixels)):
            j = isort[i]
            cls = category_ids[j]
            assert(cls >=0 and cls < num_classes)
            np.putmask(sem_seg, bitmasks[j], cls)
#             sem_seg[cls, :, :] = np.logical_or(sem_seg[cls, :, :], bitmask)
        
        return SemSegMask(torch.from_numpy(sem_seg))

#     def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
#         """
#         Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
#         This can be used to prepare training targets for Mask R-CNN.
#         It has less reconstruction error compared to rasterization with polygons.
#         However we observe no difference in accuracy,
#         but ContourMaps requires more memory to store all the masks.

#         Args:
#             boxes (Tensor): Nx4 tensor storing the boxes for each mask
#             mask_size (int): the size of the rasterized mask.

#         Returns:
#             Tensor:
#                 A bool tensor of shape (N, mask_size, mask_size), where
#                 N is the number of predicted boxes for this image.
#         """
#         assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
#         device = self.tensor.device

#         batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
#         rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

#         bit_masks = self.tensor.to(dtype=torch.float32)
#         rois = rois.to(device=device)
#         output = (
#             ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
#             .forward(bit_masks[:, None, :, :], rois)
#             .squeeze(1)
#         )
#         output = output >= 0.5
#         return output

    def get_bounding_boxes(self) -> None:
        # not needed now
        raise NotImplementedError

#     @staticmethod
#     def cat(segmasks_list: List["SemSegMask"]) -> "SemSegMask":
#         """
#         Concatenates a list of SemSegMask into a single SemSegMask

#         Arguments:
#             segmasks_list (list[SegMasks])

#         Returns:
#             SegMasks: the concatenated SegMasks
#         """
#         assert isinstance(segmasks_list, (list, tuple))
#         assert len(segmasks_list) > 0
#         assert all(isinstance(segmasks, SemSegMask) for segmasks in segmasks_list)

#         cat_segmasks = type(segmasks_list[0])(cat([cm.tensor for cm in segmasks_list], dim=0))
#         return cat_segmasks