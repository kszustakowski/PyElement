import numpy as np
import skimage
import cv2
import skimage.feature
import skimage.filters
from pykuwahara import kuwahara
from anisotropic_diffusion import anisodiff


def ensure_flattened(filter_method):
    def filter_expanded_dim(image, *args, **kwargs):
        if len(image.shape) > 2:
            image = np.squeeze(image)
        image = image.astype(np.float64)/255
        return filter_method(image, *args, **kwargs)
    return filter_expanded_dim


@ensure_flattened
def gaussian_blurs(image):
    return np.stack([
        cv2.GaussianBlur(image, (3, 3), 1),
        cv2.GaussianBlur(image, (25, 25), 3)
    ], axis=-1)


@ensure_flattened
def entropy_filters(image):
    image = skimage.util.img_as_ubyte(image)
    return np.stack([
        skimage.filters.rank.entropy(image, skimage.morphology.disk(radius=7)),
        skimage.filters.rank.entropy(image, skimage.morphology.disk(radius=15))
    ], axis=-1)


@ensure_flattened
def vesselness_filters(image):
    def vesselness_filter(scale):
        beta = 0.5
        c = 40
        hessian = skimage.feature.hessian_matrix(
            image, order='rc', sigma=scale)
        eig_1, eig_2 = skimage.feature.hessian_matrix_eigvals(hessian)
        eig_1 = eig_1.flatten()
        eig_2 = eig_2.flatten()
        eig_1_copy = np.copy(eig_1)

        eig_1[eig_1 < eig_2] = eig_2[eig_1 < eig_2]
        eig_2[eig_1_copy < eig_2] = eig_1_copy[eig_1_copy < eig_2]
        eig_2[eig_2 == 0] = float("inf")

        rb = np.power(
            np.abs(eig_1)/np.abs(eig_2), 2
        )

        eig_2[eig_2 == float("inf")] = 0

        s = np.sqrt(np.power(eig_1, 2) + np.power(eig_2, 2))

        return (np.exp(
            -np.power(rb, 2)/(2*beta*beta)
        ) * (1 - np.exp(
            -np.power(s, 2)/(2*c*c)
        ))).reshape(image.shape)
    return np.stack([
        vesselness_filter(0.6),
        vesselness_filter(0.8)
    ], axis=-1)


@ensure_flattened
def hessian_features(image):
    # Potential risk
    # For Ixy==Iyx to be true, the derivatives would have to be continous
    # Is it applicable for images?
    # Assuming yes
    Ixx, Ixy, Iyy = skimage.feature.hessian_matrix(image, order='xy')

    a = Ixx
    b = Ixy
    c = Ixy
    d = Iyy

    hessian_det = skimage.feature.hessian_matrix_det(image)

    l1, l2 = skimage.feature.hessian_matrix_eigvals(
        skimage.feature.hessian_matrix(image, order='rc')
    )

    tr_h = a + d

    modulus = np.sqrt(np.power(a, 2) + np.power(b, 2) + np.power(d, 2))

    gm_diff = np.power(a-d, 2)*(np.power(a-d, 2) + 4 * np.power(b, 2))

    return np.stack([
        a, b, c, d,
        hessian_det,
        l1, l2,
        tr_h,
        modulus,
        gm_diff
    ], axis=-1)


@ensure_flattened
def surrounding_information(image, window_size=7):
    image = (image*255).astype(np.uint8)

    median = cv2.medianBlur(image, window_size)
    mean = cv2.blur(image, (window_size, window_size))
    max_f = cv2.dilate(image, (window_size, window_size))
    min_f = cv2.erode(image, (window_size, window_size))

    median, mean, max_f, min_f = [
        o.astype(np.float64)/255 for o in (median, mean, max_f, min_f)
    ]

    return np.stack([
        median, mean,
        max_f, min_f
    ], axis=-1)


@ensure_flattened
def anisotropic_diffusion(image):

    params_set = [
        {'gamma': 0.3, 'niter': 20, 'kappa': 4},
        {'gamma': 0.5, 'niter': 10, 'kappa': 3},
        {'gamma': 2.0, 'niter': 35, 'kappa': 3},
        {'gamma': 0.8, 'niter': 40, 'kappa': 6},
    ]

    imgs = [
        anisodiff(image, **params)
        for params in params_set
    ]

    return np.stack(imgs, axis=-1).astype(np.float64)


@ensure_flattened
def morphology(image):
    B1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))*255
    B2 = skimage.morphology.disk(radius=5)*255

    morphology_config = [
        {'B': B1, 'dilations': 1, 'erosions': 1, 'type': 'opening'},
        {'B': B1, 'dilations': 2, 'erosions': 3, 'type': 'opening'},
        {'B': B2, 'dilations': 1, 'erosions': 1, 'type': 'opening'},
        {'B': B1, 'dilations': 1, 'erosions': 1, 'type': 'closing'},
        {'B': B1, 'dilations': 2, 'erosions': 3, 'type': 'closing'},
        {'B': B2, 'dilations': 1, 'erosions': 1, 'type': 'closing'},
    ]

    out = []

    for conf in morphology_config:
        source = image
        if conf['type'] == 'opening':
            source = cv2.erode(
                source, kernel=conf['B'], iterations=conf['erosions'])
            source = cv2.dilate(
                source, kernel=conf['B'], iterations=conf['dilations'])
        else:
            source = cv2.dilate(
                source, kernel=conf['B'], iterations=conf['dilations'])
            source = cv2.erode(
                source, kernel=conf['B'], iterations=conf['erosions'])
        out.append(source.astype(np.float64))

    return np.stack(out, axis=-1)


@ensure_flattened
def kuwahara_features(image):
    orig_shape = None
    if image.shape[0] != image.shape[1]:
        orig_shape = image.shape
        dim = max(orig_shape)
        image = cv2.resize(image, (dim, dim))
    kw1 = kuwahara(image, method='mean', radius=5)
    kw2 = kuwahara(image, method='mean', radius=10)
    if orig_shape:
        kw1 = cv2.resize(kw1, orig_shape)
        kw2 = cv2.resize(kw2, orig_shape)

    return np.stack([kw1, kw2], axis=-1).astype(np.float64)


@ensure_flattened
def light_sobel_filter(image, t=-10, d1=2, d2=5):
    # Rodrigues et al
    image = image*255
    b_image = cv2.copyMakeBorder(
        image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    i1 = np.logical_and(b_image - np.roll(b_image, axis=0, shift=d1) > t,
                        b_image - np.roll(b_image, axis=0, shift=-d1) > t)
    i2 = np.logical_and(b_image - np.roll(b_image, axis=1, shift=d1) > t,
                        b_image - np.roll(b_image, axis=1, shift=-d1) > t)

    LS1 = np.logical_and(i1, i2).astype(np.float64)

    i1 = np.logical_and(b_image - np.roll(b_image, axis=0, shift=d2) > t,
                        b_image - np.roll(b_image, axis=0, shift=-d2) > t)
    i2 = np.logical_and(b_image - np.roll(b_image, axis=1, shift=d2) > t,
                        b_image - np.roll(b_image, axis=1, shift=-d2) > t)

    LS2 = np.logical_and(i1, i2).astype(np.float64)

    return np.stack([LS1, LS2], axis=-1)[10:-10, 10:-10, ...]/255
