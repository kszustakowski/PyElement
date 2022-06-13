import sklearn.ensemble
import numpy as np
import cv2
import skimage
import skimage.feature
import skimage.filters
import os
import PIL.Image

class VesselClassifier(sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator):
    def __init__(self, feature_extractors, connectivity_extractors,
                 base_estimator = sklearn.ensemble.RandomForestClassifier(),
                 pca_dim=0, verbose = 0, fit_mode = "static"
                ):
        """
        Vessel Growing Classifier using the ELEMENT framework
        :param feature_extractors: - list of methods for feature extraction from the image, each method returning (image_h, image_w, n_features) shape array
        :param connectivity_extractors: - list of methods for connectivity extraction from the mask, each method returning (image_h, image_w, n_features) shape array
        :param base_estimator: - estimator to use for fitting and prediction 
        :param pca_dim: - number of target dimensions for PCA preprocessing (0 or less disables PCA)
        :param verbose: - print training info
        :param fit_mode: - 'static' for fitting using only the target features & connectivity features of the full mask
                         - 'growing' to simulate mask growing during training feature generation        
        """
        self.fit_mode = fit_mode
        self.verbose = verbose
        self.base_estimator = base_estimator
        self.feature_extractors = feature_extractors
        self.connectivity_extractors = connectivity_extractors
        self.pca_dim = pca_dim
    
    def fit(self, images_list, masks_list):
        """
        :param images_list: python list of images, each a single channel grayscale image in range 0-255 for uints or 0-1 for floats
        :param masks_list: python list of binary masks 
        """
        # Ensure images are grayscale flattened, and in uint8 0-255 range
        # Ensure masks are binary 0-1
        images_list = [self._ensure_valid_image(img) for img in images_list]
        masks_list = [self._ensure_valid_mask(mask) for mask in masks_list]
        
        image_features = [
            self.extract_image_features(img)
            for img in images_list
        ]
        
        init_masks = []
        
        for real_mask in masks_list:
            init_mask = np.zeros_like(real_mask).flatten()
            indices = np.argwhere(real_mask.flatten() > 0)
            np.random.shuffle(indices)
            init_mask[indices[:1]] = 1
            init_mask = init_mask.reshape(real_mask.shape)
            init_masks.append(init_mask)
        
        X = None
        y = None
        
        def mask_idempotent(mask, previous_mask, target_mask):
            current_trimmed = np.copy(mask)
            current_trimmed[target_mask == 0] = 0
            previous_trimmed = np.copy(previous_mask)
            previous_trimmed[target_mask == 0] = 0
            return np.array_equal(current_trimmed, previous_trimmed)
        
        if self.fit_mode == "growing":
            existing_feature_vectors = set()

            for idx, (v0, target_mask, vec_features) in enumerate(zip(init_masks, masks_list, image_features)):
                if self.verbose:
                    print("Processing input image {}".format(idx), end='')

                vi_prev = v0.astype(np.uint8)
                n_dupe = 0
                while True:
                    if self.verbose:
                        print(".", end='')
                    vi = cv2.dilate(vi_prev, np.ones((3, 3)))
                    if mask_idempotent(vi, vi_prev, target_mask):
                        break
                    to_classify = vi > vi_prev
                    vi_mask_features = np.concatenate(
                        [extractor(vi_prev) for extractor in self.connectivity_extractors],
                        axis=-1
                    )
                    feature_vector = np.concatenate([vec_features, vi_mask_features], axis=-1)
                    
                    features = feature_vector[to_classify].reshape(
                        (-1, feature_vector.shape[-1])
                    )

                    targets = target_mask[to_classify].reshape(
                        (-1,)
                    )

                    used_features = []
                    used_targets = []

                    for feature_row, target in zip(features, targets):
                        row_as_tuple = tuple(feature_row.ravel())
                        if row_as_tuple not in existing_feature_vectors:
                            used_features.append(feature_row)
                            used_targets.append(target)
                            existing_feature_vectors.add(row_as_tuple)
                        else:
                            n_dupe += 1
                    used_features = np.array(used_features)
                    used_targets = np.array(used_targets)

                    if X is None:
                        X = features
                    else:
                        X = np.append(X, features, axis=0)
                    if y is None:
                        y = targets
                    else:
                        y = np.append(y, targets, axis=0)
                    vi_prev = vi
                    vi_prev[target_mask == 0] = 0
                if self.verbose:
                    print("")
                    print("Collected ", len(X), " samples")
                    print("{} duplicate samples skipped".format(n_dupe))
        elif self.fit_mode == "static":
            for idx, (target_mask, vec_features) in enumerate(zip(masks_list, image_features)):
                mask_features = np.concatenate(
                    [extractor(target_mask) for extractor in self.connectivity_extractors],
                    axis=-1
                )
                feature_vector = np.concatenate([vec_features, mask_features], axis=-1)
                features = feature_vector.reshape(
                    (-1, feature_vector.shape[-1])
                )

                targets = target_mask.reshape(
                    (-1,)
                )
                if X is None:
                    X = features
                else:
                    X = np.append(X, features, axis=0)
                if y is None:
                    y = targets
                else:
                    y = np.append(y, targets, axis=0)
        if self.pca_dim > 0:
            if self.verbose:
                print("Fitting PCA...")
            self.pca = sklearn.decomposition.PCA(n_components=self.pca_dim)
            X = self.pca.fit_transform(X)
        if self.verbose:     
            print("Fitting classifier...")
            
        self.base_estimator.fit(X, y)
        
    def extract_image_features(self, image):
        return np.concatenate(
                [extractor(image) for extractor in self.feature_extractors],
                axis=-1
            )
        
    def predict(self, images, proba_threshold=0.5):
        """
        :param images: python list of images, each a single channel grayscale image in range 0-255 for uints or 0-1 for floats
        :param proba_threshold: threshold for the prediction probability required to classify pixel as a vessel
        """
        images_list = [self._ensure_valid_image(image) for image in images]
        masks = [self._generate_init_mask(image) for image in images]
        
        image_features = [
            self.extract_image_features(img)
            for img in images_list
        ]
        
        output_masks = []
        
        for image, image_features, mask in zip(images, image_features, masks):
            idempotent = False
            if self.verbose:
                print("Starting segmentation for a new image")
            prev_mask = mask
            while not idempotent:
                mask_features = np.concatenate(
                        [extractor(prev_mask) for extractor in self.connectivity_extractors],
                        axis=-1
                    )
                feature_vector = np.concatenate([image_features, mask_features], axis=-1)
                feature_vector = feature_vector.reshape((-1, feature_vector.shape[-1]))
                if self.pca_dim > 0:
                    feature_vector = self.pca.transform(feature_vector)
                to_predict = cv2.dilate(prev_mask, np.ones((3, 3))) - prev_mask
                new_mask = self.base_estimator.predict_proba(feature_vector)[...,1]
                new_mask = new_mask.reshape(mask.shape)
                new_mask[new_mask >= proba_threshold] = 1
                new_mask[new_mask < proba_threshold] = 0
                new_mask[to_predict == 0] = 0
                new_mask[prev_mask == 1] = 1
                if np.array_equal(new_mask, prev_mask):
                    idempotent = True
                prev_mask = new_mask
            output_masks.append(prev_mask)
        return output_masks
    
    @staticmethod
    def _generate_init_mask(image, n_seeds=1000):
        frangi_0 = skimage.filters.frangi(image)
        # Exclude values lying potentially outside of the circle
        min_dim = int(min(frangi_0.shape)*0.8/2)
        inclusion_zone = skimage.morphology.disk(radius=min_dim)
        inclusion_start_top = (frangi_0.shape[0]-inclusion_zone.shape[0])//2
        inclusion_start_left = (frangi_0.shape[1]-inclusion_zone.shape[1])//2
        inclusion_end_top = inclusion_start_top+inclusion_zone.shape[0]
        inclusion_end_left = inclusion_start_left+inclusion_zone.shape[1]

        frangi_0[:, :inclusion_start_left] = 0
        frangi_0[:inclusion_start_top, :] = 0

        frangi_0[inclusion_end_top:, :] = 0
        frangi_0[:, inclusion_end_left:] = 0

        frangi_part = frangi_0[inclusion_start_top:inclusion_end_top,
                inclusion_start_left:inclusion_end_left]
        frangi_part[inclusion_zone == 0] = 0
        frangi_0[inclusion_start_top:inclusion_end_top,
                inclusion_start_left:inclusion_end_left] = frangi_part

        frangi_shape = frangi_0.shape
        frangi_0 = frangi_0.flatten()
        max_indices=np.argpartition(frangi_0, -n_seeds, axis=None)[-n_seeds:]
        frangi_0.fill(0)
        frangi_0[max_indices] = 1
        frangi_0 = frangi_0.reshape(frangi_shape)

        return frangi_0

    
    @staticmethod
    def _ensure_valid_mask(mask):
        if len(mask.shape) > 2:
            mask = np.squeeze(mask[...,0])
        if np.amax(mask) > 1:
            mask[mask > 1] = 1
        mask = mask.astype(np.uint8)
        return mask

    @staticmethod
    def _ensure_valid_image(image):
        if len(image.shape) > 2:
            image = PIL.Image.fromarray(image)
            image = image.convert("L")
            image = np.asarray(image)
            image = np.squeeze(image)

        if np.amax(image) > 1:
            image = image*255
            image = image.astype(np.uint8)
        return image        