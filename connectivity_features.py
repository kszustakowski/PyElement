from filter_extraction import ensure_flattened
import cv2
import numpy as np
import skimage


def _mask_reduce(mask, kernel, reduction_method):
    mask = mask.copy()
    if reduction_method == 'binary':
        min_value = 1/np.sum(kernel)
        mask[mask > min_value/2] = 1
    elif reduction_method == 'normalised':
        kernel_sum = np.sum(kernel)
        mask = mask / kernel_sum
    elif reduction_method == 'none':
        pass
    else:
        raise ValueError(
            "Unsupported mask reduction method {}".format(reduction_method))
    return mask


@ensure_flattened
def immediate_connectivity(mask, reduction_method='binary'):
    immediate_mask = np.ones((3, 3))
    immediate_mask[1, 1] = 0

    n_belonging_to_mask = cv2.filter2D(
        mask, ddepth=-1, kernel=immediate_mask, borderType=cv2.BORDER_CONSTANT)
    n_belonging_to_mask = _mask_reduce(
        n_belonging_to_mask, immediate_mask, reduction_method)

    return np.stack(
        [n_belonging_to_mask],
        axis=-1
    )


@ensure_flattened
def radial_connectivity(mask, reduction_method='binary'):
    radial_mask = skimage.morphology.disk(radius=7)

    n_belonging_to_mask = cv2.filter2D(
        mask, ddepth=-1, kernel=radial_mask, borderType=cv2.BORDER_CONSTANT)
    n_belonging_to_mask = _mask_reduce(
        n_belonging_to_mask, radial_mask, reduction_method)

    return np.stack(
        [n_belonging_to_mask],
        axis=-1
    )
