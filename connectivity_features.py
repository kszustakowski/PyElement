from filter_extraction import ensure_flattened
import cv2
import numpy as np
import skimage

@ensure_flattened
def immediate_connectivity(mask):
    immediate_mask = np.ones((3, 3))
    immediate_mask[1, 1] = 0
    
    n_belonging_to_mask = cv2.filter2D(mask, ddepth=-1, kernel=immediate_mask, borderType=cv2.BORDER_CONSTANT)
    n_belonging_to_mask[n_belonging_to_mask > 0] = 1
    
    return np.stack(
        [n_belonging_to_mask],
        axis=-1
    )

@ensure_flattened
def radial_connectivity(mask):
    radial_mask = skimage.morphology.disk(radius=7)
    
    n_belonging_to_mask = cv2.filter2D(mask, ddepth=-1, kernel=radial_mask, borderType=cv2.BORDER_CONSTANT)
    min_value = 1/np.sum(radial_mask)
    n_belonging_to_mask[n_belonging_to_mask > min_value/2] = 1
    
    return np.stack(
        [n_belonging_to_mask],
        axis=-1
    )