import config as cfg
import numpy as np 


KEYPOINT_MASK_SHAPE = (cfg.image_size, cfg.image_size)
def get_keypoints_targets(positive_rois, positive_keypoints):
    
    x1 = positive_rois[:, 0]
    y1 = positive_rois[:, 1]
    x2 = positive_rois[:, 2]
    y2 = positive_rois[:, 3]

    scale_x = KEYPOINT_MASK_SHAPE[1] / (x2 - x1)
    scale_y = KEYPOINT_MASK_SHAPE[0] / (y2 - y1)

    keypoints_targets = []
    keypoints_weights = []
    for k in range(cfg.keypoints_num):
        kx1 = positive_keypoints[:, k, 0]
        ky1 = positive_keypoints[:, k, 1]

        # print(kx1 - x1 + 0.5)
        kx1_map = (((kx1 - x1) + 0.5) * scale_x).astype(np.int32)
        ky1_map = (((ky1 - y1) + 0.5) * scale_y).astype(np.int32)
        x_boundary_bool = (kx1_map == KEYPOINT_MASK_SHAPE[1]).astype(np.int32)
        y_boundary_bool = (ky1_map == KEYPOINT_MASK_SHAPE[0]).astype(np.int32)
        ky1_map = ky1_map * (1 - y_boundary_bool) + y_boundary_bool * (KEYPOINT_MASK_SHAPE[1] - 1)
        kx1_map = kx1_map * (1 - x_boundary_bool) + x_boundary_bool * (KEYPOINT_MASK_SHAPE[0] - 1)
        

        valid = np.logical_and(
            np.logical_and(kx1_map > 0, kx1_map < KEYPOINT_MASK_SHAPE[0]),
            np.logical_and(ky1_map > 0, ky1_map < KEYPOINT_MASK_SHAPE[1])
        )
        # print(valid)
        keypoints_weights.append(valid)
        
        keypoints_target = ky1_map * KEYPOINT_MASK_SHAPE[1] + kx1_map
        # print(keypoints_target)
        keypoints_targets.append(keypoints_target)
    
    # print(keypoints_targets)
    # print(keppoints_weights)
    keypoints_targets = np.stack(keypoints_targets, axis=1)
    keypoints_weights = np.stack(keypoints_weights, axis=1)
    keypoints_targets[np.where(keypoints_weights==0)] = -1
    # keypoints_weights_modified = keypoints_weights * np.array(cfg.keypoints_weights, dtype=np.float32)
    # print(keypoints_weights.shape, keypoints_targets.shape)
    return keypoints_targets, keypoints_weights