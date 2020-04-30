import numpy as np
import cv2

def map_coco_to_personlab(keypoints):
    permute = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    if len(keypoints.shape) == 2:
        return keypoints[permute, :]
    return keypoints[:, permute, :]

   
# List of edges as tuples of indices into the KEYPOINTS array
# (Each edge will be used twice in the mid-range offsets; once in each direction)
EDGES = [
    (0, 14),
    (0, 13),
    (0, 4),
    (0, 1),
    (14, 16),
    (13, 15),
    (4, 10),
    (1, 7),
    (10, 11),
    (7, 8),
    (11, 12),
    (8, 9),
    (4, 5),
    (1, 2),
    (5, 6),
    (2, 3)
]


class KeypointMaps:
    """
    Main class for keypoint-based distance maps.
    """
    
    def __init__(self, all_keypoints, map_shape, n_kp=17, kp_radius=32, edges=EDGES):
        self.all_kpts = all_keypoints
        self.map_shape = map_shape
        self.idx = np.rollaxis(np.indices(self.map_shape[::-1]), 0, 3).transpose((1,0,2))
        self.n_inst = len(all_keypoints)
        
        self.n_kp = n_kp
        self.kp_radius = kp_radius
        
        self.edges = edges
        self.n_edges = len(self.edges)
        
    
    # return (人数,key数,401,401) 绘制heatmap的响应圆 
    def get_keypoint_discs(self):
        discs = [[] for _ in range(self.n_inst)]
        for i in range(self.n_kp):
            centers = [keypoints[i,:2] for keypoints in self.all_kpts if keypoints[i,2] > 0]
            dists = np.zeros(self.map_shape+(len(centers),))

            for k, center in enumerate(centers):
                dists[:,:,k] = np.sqrt(np.square(center-self.idx).sum(axis=-1))
            if len(centers) > 0:
                inst_id = dists.argmin(axis=-1)
            count = 0
            for j in range(self.n_inst):
                if self.all_kpts[j][i,2] > 0:
                    discs[j].append(np.logical_and(inst_id==count, dists[:,:,count]<=self.kp_radius))
                    count +=1
                else:
                    discs[j].append(np.array([]))
        return discs

    
    def make_keypoint_maps(self, discs):
        kp_maps = np.zeros(self.map_shape+(self.n_kp,))
        for i in range(self.n_kp):
            for j in range(len(discs)):
                if self.all_kpts[j][i,2] > 0:
                    kp_maps[discs[j][i], i] = 1.

        return kp_maps


    def compute_short_offsets(self, discs):
        r = self.kp_radius
        x = np.tile(np.arange(r, -r-1, -1), [2*r+1, 1])
        y = x.transpose()
        m = np.sqrt(x*x + y*y) <= r
        kp_circle = np.stack([x, y], axis=-1) * np.expand_dims(m, axis=-1)

        def copy_with_border_check(maps, center, disc):
            from_top = max(r-center[1], 0)
            from_left = max(r-center[0], 0)
            from_bottom = max(r-(self.map_shape[0]-center[1])+1, 0)
            from_right =  max(r-(self.map_shape[1]-center[0])+1, 0)

            cropped_disc = disc[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right]
            maps[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right, :][cropped_disc,:] = \
                                        kp_circle[from_top:2*r+1-from_bottom, from_left:2*r+1-from_right, :][cropped_disc,:]

        offsets = np.zeros(self.map_shape+(2*self.n_kp,))
        for i in range(self.n_kp):
            for j in range(self.n_inst):
                if self.all_kpts[j][i,2] > 0:
                    copy_with_border_check(offsets[:,:,2*i:2*i+2], (self.all_kpts[j][i,0].astype(np.int), self.all_kpts[j][i,1].astype(np.int)), discs[j][i].astype(np.int))

        return offsets


    def compute_mid_offsets(self, discs):
        offsets = np.zeros(self.map_shape+(4*self.n_edges,))
        for i, edge in enumerate((self.edges + [edge[::-1] for edge in self.edges])):
            for j in range(self.n_inst):
                if self.all_kpts[j][edge[0],2] > 0 and self.all_kpts[j][edge[1],2] > 0:
                    m = discs[j][edge[0]]
                    dists = [[ self.all_kpts[j][edge[1],0], self.all_kpts[j][edge[1],1] ]] - self.idx[m,:]
                    offsets[m,2*i:2*i+2] = dists

        return offsets


    def compute_long_offsets(self, instance_masks):
        instance_masks = instance_masks.astype('bool')
        offsets = np.zeros(self.map_shape+(2*self.n_kp,))
        for i in range(self.n_kp):
            for j in range(self.n_inst):
                if self.all_kpts[j][i,2] > 0:
                    m = instance_masks[:,:,j]
                    dists = [[ self.all_kpts[j][i,0], self.all_kpts[j][i,1] ]] - self.idx[m,:]
                    offsets[m, 2*i:2*i+2] = dists

        overlap = np.sum(instance_masks, axis=-1) >= 2
        offsets[overlap,:] = 0.
        return offsets

    
def get_keypoint_maps(instance_masks, all_keypoints, map_shape, n_kp = 17):
    # assert(instance_masks.shape[-1] == len(all_keypoints))
    km = KeypointMaps(all_keypoints, map_shape, n_kp = 17)
    discs = km.get_keypoint_discs()
    kp_maps = km.make_keypoint_maps(discs)
    short_offsets = km.compute_short_offsets(discs)
#     mid_offsets = compute_mid_offsets(all_keypoints, discs)
#     long_offsets = compute_long_offsets(all_keypoints, instance_masks)

    return kp_maps, short_offsets  # , mid_offsets, long_offsets
