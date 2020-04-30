# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn import functional as F
import torch

from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances, Boxes

import numpy as np
import cv2


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = paste_masks_in_image(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result


def seg_det_postprocess_bk(segmap, contour, emb, img_size, output_height, output_width):
    """
    Translate segmentation predictions into detection results.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        segmap (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        contour (Tensor): contour prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        emb (Tensor): contour prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    segmap = segmap[:, : img_size[0], : img_size[1]].cpu().numpy()
    contour = contour[:, : img_size[0], : img_size[1]].cpu().numpy()
    emb = emb[:, : img_size[0], : img_size[1]].cpu().numpy()
        
    ncls = segmap.shape[0] - 1  # remove the background
    assert(contour.shape[0]==ncls)
    assert(emb.shape[0]==ncls)
    H = segmap.shape[1]
    W = segmap.shape[2]
    
    pred_boxes = []
    pred_scores = []
    pred_classes = []
    pred_masks = []
    # Step1: segment the foreground (according to segmap) into super-pixels (separated by contours)
    for c in range(ncls):
        cont_c = contour[c]
        seg_c = segmap[c]
        emb_c = emb[c]
        #TODO: we may need to turn the contour map and segmap into binary images
        #For now we combine contour map and segmentation before connecting superpixels for simplicity
        bw = (1-cont_c) * seg_c > 0.2  # 0.05 # 1-cont_c > 0.7  # 
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bw.astype(np.uint8))
        nseg = retval  # np.max(labels)
        # Note: the background is labeled to b 0, which should be ignored
#         assert(retval==nseg+1)
        avg_embed = np.zeros(nseg)
        avg_scores = np.zeros(nseg)
        bboxes = np.zeros((nseg, 4))
        for s in range(nseg):
            seg_size = stats[s, cv2.CC_STAT_AREA]
            if seg_size<H*W*0.0001:  # 0.008 0.0002
                continue
            
            # calculate the average embedding of each superpixel
            superpixel = labels==s
            superpixel = superpixel.astype(np.float)
            npixel = np.sum(superpixel)
            avg_scores[s] = np.sum(seg_c * superpixel) / npixel
            avg_embed[s] = np.sum(emb_c * superpixel) / npixel
            
            # get the bounding boxes of superpixels in X1Y1X2Y2 format
            bboxes[s, 0] = stats[s, cv2.CC_STAT_LEFT]
            bboxes[s, 1] = stats[s, cv2.CC_STAT_TOP]
            bboxes[s, 2] = stats[s, cv2.CC_STAT_WIDTH] + bboxes[s, 0]
            bboxes[s, 3] = stats[s, cv2.CC_STAT_HEIGHT] + bboxes[s, 1]
        
        # remove small segments and low-confident segments
        idx = [s for s in range(nseg) if avg_scores[s]>=0.2]  # 0.2
        avg_embed = avg_embed[idx]
        avg_scores = avg_scores[idx]
        bboxes = bboxes[idx, :]
        nseg = len(avg_scores)
        
        # Step 2: group the super-pixels of the same object according to embedding
        bmerged = np.zeros(nseg, dtype=np.bool)
        merged_bboxes = bboxes
        areas = np.zeros(nseg, dtype=np.float)
        for s in range(nseg):
            if bmerged[s]:
                continue
            areas[s] = (merged_bboxes[s, 3]-merged_bboxes[s, 1])*(merged_bboxes[s, 2]-merged_bboxes[s, 0])
            for t in range(s+1, nseg):
                # TODO: we may take spatial distance as an auxiliary criteria
                if abs(avg_embed[s]-avg_embed[t])<0.2:  # 0.2
#                     idx = np.where(labels==t)
#                     labels[idx] = s
                    
                    # merge the bounding boxes
                    merged_bboxes[s, 0] = min(merged_bboxes[s, 0], merged_bboxes[t, 0])
                    merged_bboxes[s, 1] = min(merged_bboxes[s, 1], merged_bboxes[t, 1])
                    merged_bboxes[s, 2] = max(merged_bboxes[s, 2], merged_bboxes[t, 2])
                    merged_bboxes[s, 3] = max(merged_bboxes[s, 3], merged_bboxes[t, 3])
                    
                    areas[s] = (merged_bboxes[s, 3]-merged_bboxes[s, 1])*(merged_bboxes[s, 2]-merged_bboxes[s, 0])
                    
                    # merge the scores
                    avg_scores[s] = max(avg_scores[s], avg_scores[t])
                    
                    bmerged[t] = True
        
        ileft = [s for s in range(nseg) if not bmerged[s] and areas[s]>H*W*0.002]
        avg_scores = avg_scores[ileft]
        bboxes = merged_bboxes[ileft, :].astype(np.int32)
        nseg = len(avg_scores)
        masks = []
        for i in range(nseg):
            mask = np.zeros(img_size, dtype=np.float)
            mask[bboxes[i,0]:bboxes[i,2],bboxes[i,1]:bboxes[i,3]] = seg_c[bboxes[i,0]:bboxes[i,2],bboxes[i,1]:bboxes[i,3]]
            masks.append(mask)
        
        pred_boxes.append(bboxes)
        pred_scores.append(avg_scores)
        pred_classes += [c]*len(avg_scores)
        pred_masks += masks
        

    # rescale the bounding boxes to match the output resolution
    scale_x, scale_y = (output_width / img_size[1], output_height / img_size[0])
    result = Instances((output_height, output_width))  # img_size   
    output_boxes = Boxes(torch.tensor(np.concatenate(pred_boxes).astype(int)))
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(result.image_size)
    result.pred_boxes = output_boxes
#     output_boxes.clip(results.image_size)
    result.scores = torch.tensor(np.concatenate(pred_scores))
    result.pred_classes = torch.tensor(pred_classes)
#     result.pred_masks = torch.tensor(np.concatenate(pred_masks))  #TODO: we have to rescale to the output size
    
    return result


def seg_det_postprocess(segmap, contour, emb, img_size, output_height, output_width):
    """
    Translate segmentation predictions into detection results.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        segmap (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        contour (Tensor): contour prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        emb (Tensor): contour prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    segmap = segmap[:, : img_size[0], : img_size[1]].cpu().numpy()
    contour = contour[:, : img_size[0], : img_size[1]].cpu().numpy()
    emb = emb[:, : img_size[0], : img_size[1]].cpu().numpy()
        
    ncls = segmap.shape[0] - 1  # remove the background
    assert(contour.shape[0]==ncls)
    assert(emb.shape[0]==ncls)
    H = segmap.shape[1]
    W = segmap.shape[2]
    
    pred_boxes = []
    pred_scores = []
    pred_classes = []
    pred_masks = []
    # Step1: segment the foreground (according to segmap) into super-pixels (separated by contours)
    for c in range(ncls):
        cont_c = contour[c]
        seg_c = segmap[c]
        emb_c = emb[c]
        #TODO: we may need to turn the contour map and segmap into binary images
        #For now we combine contour map and segmentation before connecting superpixels for simplicity
        bw = (1-1.5*cont_c) * seg_c > 0.2  # 0.05 # 1-cont_c > 0.5   # 
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bw.astype(np.uint8))
        nseg = np.max(labels)
        # Note: the background is labeled to b 0, which should be ignored
        assert(retval==nseg+1)
        avg_embed = np.zeros(nseg)
        avg_scores = np.zeros(nseg)
        bboxes = np.zeros((nseg, 4))
        bboxes_size = np.zeros(nseg)
        sizes = np.zeros(nseg)
        for s in range(nseg):
            sizes[s] = stats[s+1, cv2.CC_STAT_AREA]
            if sizes[s]<H*W*0.0001:
                continue

            # calculate the average embedding of each superpixel
            superpixel = labels==s+1
            superpixel = superpixel.astype(np.float)
            npixel = np.sum(superpixel)
            avg_scores[s] = np.sum(seg_c * superpixel) / npixel
        #     avg_embed[s] = np.sum(emb_c * superpixel) / npixel

            # calculate the median value of the segment
            ipixels = np.nonzero(labels==s+1)
            avg_embed[s] = np.median(emb_c[ipixels])
        #     print(avg_embed[s] - median_embed)

            # get the bounding boxes of superpixels in X1Y1X2Y2 format
            bboxes[s, 0] = stats[s+1, cv2.CC_STAT_LEFT]
            bboxes[s, 1] = stats[s+1, cv2.CC_STAT_TOP]
            bboxes[s, 2] = stats[s+1, cv2.CC_STAT_WIDTH] + bboxes[s, 0]
            bboxes[s, 3] = stats[s+1, cv2.CC_STAT_HEIGHT] + bboxes[s, 1]
            bboxes_size[s] = (bboxes[s, 3]-bboxes[s, 1])*(bboxes[s, 2]-bboxes[s, 0])

        # Step 2: remove low-confident segments
        idx = [s for s in range(nseg) if avg_scores[s]>=0.2]  # 0.05
        avg_embed = avg_embed[idx]
        avg_scores = avg_scores[idx]
        bboxes = bboxes[idx, :]
        bboxes_size = bboxes_size[idx]
        sizes = sizes[idx]
        nseg = len(avg_scores)

        # Step 3: Sort the segments by size
        sorted_idx = np.flip(np.argsort(sizes))
        avg_embed = avg_embed[sorted_idx]
        avg_scores = avg_scores[sorted_idx]
        bboxes = bboxes[sorted_idx, :]
        bboxes_size = bboxes_size[sorted_idx]
        sizes = sizes[sorted_idx]

        # Step 4: calculate the similarity between each pair of segments
        sim = np.zeros((nseg, nseg))
        if nseg >=2:
            emb_sigma = avg_embed.std()
        else:
            emb_sigma = 0.5

        SIM_EMB_FACTOR = 0.8  # 1.5 5*(avg_embed.max()-avg_embed.min())/nseg   # 1.0/(emb_sigma*np.sqrt(2*np.pi))
        for s in range(nseg):
            for t in range(s+1, nseg):
                # similarity of embedding
                #sim_emb = np.exp(-SIM_EMB_FACTOR * np.abs(avg_embed[s]-avg_embed[t])/emb_var)
                #sim_emb = SIM_EMB_FACTOR * np.exp(-np.math.pow(avg_embed[s]-avg_embed[t], 2)/(2*2*emb_sigma**2))
                sim_emb = np.exp(-np.abs(avg_embed[s]-avg_embed[t])/SIM_EMB_FACTOR)
        #         sim_emb = np.exp(-4*np.abs(avg_embed[s]-avg_embed[t])/(np.abs(avg_embed[s])+np.abs(avg_embed[t])))

                # spatial distance based on GIOU
                merged_bbox = np.zeros(4)
                merged_bbox[0] = min(bboxes[s, 0], bboxes[t, 0])
                merged_bbox[1] = min(bboxes[s, 1], bboxes[t, 1])
                merged_bbox[2] = max(bboxes[s, 2], bboxes[t, 2])
                merged_bbox[3] = max(bboxes[s, 3], bboxes[t, 3])

                merged_area = (merged_bbox[3]-merged_bbox[1])*(merged_bbox[2]-merged_bbox[0])

                overlap_bbox = np.zeros(4)
                overlap_bbox[0] = max(bboxes[s, 0], bboxes[t, 0])
                overlap_bbox[1] = max(bboxes[s, 1], bboxes[t, 1])
                overlap_bbox[2] = min(bboxes[s, 2], bboxes[t, 2])
                overlap_bbox[3] = min(bboxes[s, 3], bboxes[t, 3])

                overlap_area = max(0, overlap_bbox[2]-overlap_bbox[0])*max(0, overlap_bbox[3]-overlap_bbox[1])

                sim_spatial = (bboxes_size[s]+bboxes_size[t]-overlap_area) / merged_area

                #TODO: calculate contour-based distance
                
                sim[s, t] = sim_spatial * sim_emb  # 
                sim[t, s] = sim[s, t]

        #TODO: calculate the keypoint-based similarity
        
        
        # Step 5: group the segments of the same object according to the similarity matrix
        bmerged = np.zeros(nseg, dtype=np.bool)
        group_IDs = np.ones(nseg, dtype=np.int) * -1
        ngroups = 0
        THR_SIM = 0.5
        while any(group_IDs < 0):
            for s in range(nseg):
                if group_IDs[s] < 0:
                    # find out the closest segment
                    assigned = np.nonzero(group_IDs >= 0)
                    if assigned[0].size == 0:
                        group_IDs[s] = ngroups
                        ngroups += 1
                    else:
                        sim_group = sim[s, assigned[0]]
                        t = np.argmax(sim_group)
        #                 print(sim_group)
        #                 print(t)
                        if sim_group[t] > THR_SIM:
                            group_IDs[s] = group_IDs[assigned[0][t]]
                        else:
                            group_IDs[s] = ngroups
                            ngroups += 1

        # merge the groups
        group_bboxes = np.zeros((ngroups, 4))
        group_scores = np.zeros(ngroups)
        group_areas = np.zeros(ngroups)
        for g in range(ngroups):
            assigned = np.nonzero(group_IDs == g)
            assigned = assigned[0]
            group_bboxes[g, :] = bboxes[assigned[0], :]
            group_scores[g] = avg_scores[assigned[0]]
            group_areas[g] = sizes[assigned[0]]
            for s in range(1, len(assigned)):
                # merge the bounding boxes
                group_bboxes[g, 0] = min(bboxes[assigned[s], 0], group_bboxes[g, 0])
                group_bboxes[g, 1] = min(bboxes[assigned[s], 1], group_bboxes[g, 1])
                group_bboxes[g, 2] = max(bboxes[assigned[s], 2], group_bboxes[g, 2])
                group_bboxes[g, 3] = max(bboxes[assigned[s], 3], group_bboxes[g, 3])

                # areas[s] = (merged_bboxes[s, 3]-merged_bboxes[s, 1])*(merged_bboxes[s, 2]-merged_bboxes[s, 0])
                group_areas[g] += sizes[assigned[s]]

#                 # merge the scores # use the score of the large segment
#                 group_scores[g] = max(avg_scores[assigned[s]], group_scores[g])

        if nseg:
            THR_AREA = max(sizes[0]*0.1, H*W*0.001)
        else:
            THR_AREA = H*W*0.001  # 0.002
        ileft = np.nonzero(group_areas>THR_AREA)
        ileft = ileft[0]
        avg_scores = group_scores[ileft]
        bboxes = group_bboxes[ileft, :].astype(np.int32)
        nseg = len(avg_scores)
        masks = []
        for i in range(nseg):
            mask = np.zeros((H, W), dtype=np.float)
            mask[bboxes[i,0]:bboxes[i,2],bboxes[i,1]:bboxes[i,3]] = seg_c[bboxes[i,0]:bboxes[i,2],bboxes[i,1]:bboxes[i,3]]
            masks.append(mask)

        pred_boxes.append(bboxes)
        pred_scores.append(avg_scores)
        pred_classes += [c]*len(avg_scores)
        pred_masks += masks
        

    # rescale the bounding boxes to match the output resolution
    scale_x, scale_y = (output_width / img_size[1], output_height / img_size[0])
    result = Instances((output_height, output_width))  # img_size   
    output_boxes = Boxes(torch.tensor(np.concatenate(pred_boxes).astype(int)))
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(result.image_size)
    result.pred_boxes = output_boxes
#     output_boxes.clip(results.image_size)
    result.scores = torch.tensor(np.concatenate(pred_scores))
    result.pred_classes = torch.tensor(pred_classes)
#     result.pred_masks = torch.tensor(np.concatenate(pred_masks))  #TODO: we have to rescale to the output size
    
    return result