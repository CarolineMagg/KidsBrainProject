########################################################################################################################
# Class to calculate SVM features of image and segmentation mask
########################################################################################################################

import numpy as np
import cv2
import os
import sys
import pandas as pd
from scipy.spatial.distance import cdist
import logging as log

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class SVMFeatures:
    """
    Class to calculate the independent (shape and appearance) features and the
    dependent (error metrics) features for training a SVM with images, groundtruth and predicted segmentation masks.
    """

    def __init__(self, imgs, gts, preds, k=1):
        """
        Constructor
        :param imgs: list of images
        :param gts: list of ground truth segmentation masks
        :param preds: list of predicted segmentation masks
        :param k: value of segmentation (default: 1)
        """
        self.img_list = imgs
        self.gt_list = gts
        self.pred_list = preds
        self.segm_value = k

    def calculate(self, standardize=True):
        """
        Calculates the independent and dependent features
        :return: dicts for independent and dependent features per image
        """
        indep_features = self.calculate_independent_features()
        dep_features = self.calculate_dependent_features()
        for idx in range(len(indep_features)):
            if np.isnan(np.sum(np.array(indep_features.loc[idx]))):  # if one feature couldn't be calculated, remove row
                indep_features = indep_features.drop(index=idx)
                dep_features = dep_features.drop(index=idx)
        assert len(indep_features) == len(dep_features), "SVMFeatures: not same number of independent and dependent features."
        if standardize:
            indep_features = self.standardize(indep_features)
            dep_features = self.standardize(dep_features)
        return indep_features, dep_features

    @staticmethod
    def standardize(df):
        for col in df:
            average = np.mean(df[col])
            std = np.std(df[col])
            df[col] = df[col].apply(lambda x: (x - average) / std if std != 0 else x)
        return df

    def calculate_independent_features(self):
        """
        Calculates the independent features for SVM
        * weighted and unweighted geometry features
        * intensity features
        * gradient features
        :return: dict with all features per image
        """
        # unweighted geometry features
        log.info(" unweighted geometry features")
        unweigted_geometry = self._calc_unweighted_geometry(self.pred_list, self.segm_value)
        # weighted geometry features
        log.info(" weighted geometry features")
        weighted_geometry = self._calc_weighted_geometry(self.img_list, self.pred_list, self.segm_value)
        # intensity features
        log.info(" intensity features")
        intensity = self._calc_intensity(self.img_list, self.pred_list, self.segm_value)
        # gradient features
        log.info(" gradient features")
        gradients = self._calc_gradients(self.img_list, self.pred_list, self.segm_value)
        # ratio features
        log.info(" ratio features")
        ratios = self._calc_ratios(unweigted_geometry, weighted_geometry, gradients)
        # concatenate results
        assert len(unweigted_geometry) == len(weighted_geometry) == len(intensity) == len(gradients)
        return unweigted_geometry.join([weighted_geometry, intensity, gradients, ratios])

    def calculate_dependent_features(self):
        """
        Calculates the dependent features (error metrics) for SVM
        * dice coefficient
        * jaccard distance
        * hausdorff distance
        * average surface error
        :return: dict with all error metrics per image
        """
        # dice coefficient
        result = pd.DataFrame(index=range(len(self.img_list)), columns=['dice_coeff', 'jaccard_dist', 'hausdorff_dist',
                                                                        'mod_hausdorff_dist', 'avg_surface_error'])
        log.info(" dice coefficient")
        result['dice_coeff'] = SVMFeatures.dice_coeff(self.gt_list, self.pred_list, self.segm_value)
        # jaccard distance
        log.info(" jaccard distance")
        result['jaccard_dist'] = SVMFeatures.jaccard_distance(self.gt_list, self.pred_list, self.segm_value)
        # hausdorff distance
        result['hausdorff_dist'] = SVMFeatures.hausdorff_distance(self.gt_list, self.pred_list, self.segm_value)
        # modified hausdorff distance
        log.info(" modified hausdorff distance")
        result['mod_hausdorff_dist'] = SVMFeatures.mod_hausdorff_distance(self.gt_list, self.pred_list, self.segm_value)
        # average surface error
        log.info(" average surface error")
        result['avg_surface_error'] = SVMFeatures.average_surface_error(self.gt_list, self.pred_list, self.segm_value)
        return result

    @staticmethod
    def _calc_unweighted_geometry(preds, k=1):
        """
        Calculates the 2 unweighted geometry features:
        * volume (number of pixels in segmentation prediction)
        * surface area (number of pixels on segmentation prediction contour)
        :param preds: list of segmentation masks
        :param k: value of segmentation (default: 1)
        :return: dict with 2 unweighted geometry features per prediction
        """
        result = pd.DataFrame(index=range(len(preds)), columns=['volume', 'surface_area'])
        for idx, pred in enumerate(preds):
            if ~(pred == k).any():  # no prediction
                continue
            # volume
            volume = np.sum(pred == k)
            # surface area
            im3, contours, hierarchy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            processed = cv2.drawContours(np.zeros_like(im3), contours, -1, k, 1)
            surface_area = np.sum(processed == k)
            # total curvature
            #mean_curvature = np.sum(SVMFeatures._calc_curvature(pred, k))
            # results
            result.loc[idx] = [volume, surface_area]
        return result

    @staticmethod
    def _calc_curvature(pred, k=1):
        """
        Calculates mean curvature of prediction boundary
        :param pred: segmentation prediction
        :return: mean curvature of boundary
        """
        im3, contours, hierarchy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        processed = cv2.drawContours(np.zeros_like(im3), contours, -1, k, 1)
        dx, dy = np.gradient(processed)
        d2x, dxy = np.gradient(dx)
        dyx, d2y = np.gradient(dy)
        mean_curvature = np.abs(d2x * dy - dx * d2y) / ((dx * dx + dy * dy) ** 1.5)
        mean_curvature[np.isnan(mean_curvature)] = 0
        return mean_curvature

    @staticmethod
    def _calc_weighted_geometry(imgs, preds, k=1):
        """
        Calculates the 4 weighted geometry features:
        * weighted volume (sum over the weights of all pixels inside the segmentation)
        * weighted cut (sum over the weights of pixels at segmentation boundary)
        * low-hi weighted cut (sum over all outgoing weights of pixels at segmentation boundary)
        * hi-low weighted cut (sum over all incoming weights of pixels at segmentation boundary)
        :param imgs: list of images
        :param preds: list of segmentation masks
        :param k: value of segmentation (default: 1)
        :return: dict with 4 weighted geometry features per image
        """
        assert len(imgs) == len(preds), 'SVMFeatures: images and predictions not the same length.'
        result = pd.DataFrame(index=range(len(imgs)), columns=['weighted_volume', 'weighted_cut',
                                                               'lh_weighted_cut', 'hl_weighted_cut'])
        for idx, img, pred in zip(range(len(imgs)), imgs, preds):
            if ~(pred == k).any():  # no prediction
                continue
            M = max((np.abs(np.gradient(img)[0]) + np.abs(np.gradient(img)[1]))[pred == k])
            # weighted volume
            weighted_volume = SVMFeatures.calc_weighted_volume(img, pred, M)
            # weighted cut
            weighted_cut = SVMFeatures.calc_weighted_cut(img, pred, M)
            # low-hi weighted cut
            lh_weighted_cut = SVMFeatures.calc_low_hi_weighted_cut(img, pred, M)
            # hi-low weighted cut
            hl_weighted_cut = SVMFeatures.calc_hi_low_weighted_cut(img, pred, M)
            # weighted curvature
            #weighted_curvature = SVMFeatures.calc_weighted_curvature(img, pred, M, k)
            # result
            result.loc[idx] = [weighted_volume, weighted_cut, lh_weighted_cut, hl_weighted_cut]
        return result

    @staticmethod
    def _calc_intensity(imgs, preds, k=1):
        """
        Calculates the 7 intensity features:
        * mean, median intensity of image within segmentation
        * sum of image intensity within segmentation
        * min, max of image intensity within segmentation
        * interquartile distance (half of the difference between 75th and 25th percentile value)
        * standard deviation of image intensity within segmentation
        :param imgs: list of images
        :param preds: list of segmentation masks
        :param k: value of segmentation (default: 1)
        :return: dict with 7 intensity features per image
        """
        assert len(imgs) == len(preds), 'SVMFeatures: images and predictions not the same length.'
        result = pd.DataFrame(index=range(len(imgs)), columns=['mean_intensity', 'median_intensity',
                                                               'sum_intensity', 'min_intensity', 'max_intensity',
                                                               'iqr_distance', 'std'])
        for idx, img, pred in zip(range(len(imgs)), imgs, preds):
            if ~(pred == k).any():  # no prediction
                continue
            # mean
            mean_intensity = np.mean(img[pred == k])
            # median
            median_intensity = np.median(img[pred == k])
            # sum
            sum_intensity = np.sum(img[pred == k])
            # min
            min_intensity = np.min(img[pred == k])
            # max
            max_intensity = np.max(img[pred == k])
            # interquartile distance
            iqr_distance = np.subtract(*np.percentile(img[pred == k], [75, 25])) / 2
            # standard deviation
            std = np.std(img[pred == k])
            result.loc[idx] = [mean_intensity, median_intensity, sum_intensity, min_intensity, max_intensity,
                               iqr_distance, std]
        return result

    @staticmethod
    def _calc_gradients(imgs, preds, k=1):
        """
        Calculates the 10 gradient features:
        * sum of L1 norm of gradients within segmentation
        * mean, median of L1 norm of gradients within segmentation
        * min, max of L1 norm of gradients within segmentation
        * standard deviation L1 norm of gradients within segmentation
        * interquartile range L1 norm of gradients within segmentation
        * sum of L2 norm of gradients withing segmentation
        * mean of L2 norm of gradients within segmentation
        * standard deviation of L2 norm of gradients within segmentation
        :param imgs: list of images
        :param preds: list of segmentation masks
        :param k: value of segmentation (default: 1)
        :return: dict with 10 gradient features per image
        """
        assert len(imgs) == len(preds), 'SVMFeatures: images and predictions not the same length.'
        result = pd.DataFrame(index=range(len(imgs)), columns=['sum_l1', 'sum_l2',
                                                               'mean_l1', 'mean_l2',
                                                               'std_l1', 'std_l2',
                                                               'median_l1', 'min_l1', 'max_l1',
                                                               'iqr_l1'])
        for idx, img, pred in zip(range(len(imgs)), imgs, preds):
            if ~(pred == k).any():  # no prediction
                continue
            gradients_l1 = (np.abs(np.gradient(img)[0]) + np.abs(np.gradient(img)[1]))[pred == k]
            gradients_l2 = (np.sqrt((np.gradient(img)[0]) ** 2 + (np.gradient(img)[1]) ** 2))[pred == k]
            # sum L1 norm
            sum_l1 = np.sum(gradients_l1)
            # mean L1 norm
            mean_l1 = np.mean(gradients_l1)
            # median L1 norm
            median_l1 = np.median(gradients_l1)
            # min L1 norm
            min_l1 = np.min(gradients_l1)
            # max L1 norm
            max_l1 = np.max(gradients_l1)
            # std L1 norm
            std_l1 = np.std(gradients_l1)
            # interquartile distance
            iqr_l1 = np.subtract(*np.percentile(gradients_l1, [75, 25])) / 2
            # sum L2 norm
            sum_l2 = np.sum(gradients_l2)
            # mean L2 norm
            mean_l2 = np.mean(gradients_l2)
            # std L2 norm
            std_l2 = np.std(gradients_l2)
            result.loc[idx] = [sum_l1, sum_l2,
                               mean_l1, mean_l2,
                               std_l1, std_l2,
                               median_l1, min_l1, max_l1,
                               iqr_l1]
        return result

    def _calc_ratios(self, unweigthed_features, weighted_features, gradients):
        """
        Calculates 11 ratio features from previous weighted, unweighted geometry and gradient features
        * blur index (sum l2 /sum l1)
        * low-hi weighted cut divided by weighted/unweighted volume
        * hi-low weighted cut divided by weighted/unweighted volume
        * low-hi weighted cut divided by weighted/unweighted cut
        * hi-low weighted cut divided by weighted/unweighted cut
        * weighted cut divided by unweighted cut
        * weighted cut divided by volume
        * unweighted cut divided by volume
        Note: unweighted cut is surface area
        :param unweigthed_features: unweighted geometry features
        :param weighted_features: weighted geoemtry features
        :param gradients: gradient features
        :return: dict with 11 ratio features
        """
        result = pd.DataFrame(columns=['blur_index_ratio',
                                       'lh_volume_ratio', 'lh_weighted_volume_ratio',
                                       'hl_volume_ratio', 'hl_weighted_volume_ratio',
                                       'lh_weighted_cut_ratio', 'lh_surface_area_ratio',
                                       'hl_weighted_cut_ratio', 'hl_surface_area_ratio',
                                       'weighted_cut_unweighted_cut_ratio',
                                       'weighted_cut_volume_ratio', 'surface_area_volume_ratio'])
        assert len(unweigthed_features) == len(weighted_features) == len(gradients)
        result['blur_index_ratio'] = gradients['sum_l2']/gradients['sum_l1']
        result['lh_volume_ratio'] = weighted_features['lh_weighted_cut']/unweigthed_features['volume']
        result['lh_weighted_volume_ratio'] = weighted_features['lh_weighted_cut']/weighted_features['weighted_volume']
        result['hl_volume_ratio'] = weighted_features['hl_weighted_cut'] / unweigthed_features['volume']
        result['hl_weighted_volume_ratio'] = weighted_features['hl_weighted_cut'] / weighted_features['weighted_volume']
        result['lh_weighted_cut_ratio'] = weighted_features['lh_weighted_cut']/weighted_features['weighted_cut']
        result['lh_surface_area_ratio'] = weighted_features['lh_weighted_cut']/unweigthed_features['surface_area']
        result['hl_weighted_cut_ratio'] = weighted_features['hl_weighted_cut']/weighted_features['weighted_cut']
        result['hl_surface_area_ratio'] = weighted_features['hl_weighted_cut']/unweigthed_features['surface_area']
        result['weighted_cut_unweighted_cut_ratio'] = weighted_features['weighted_cut']/unweigthed_features['surface_area']
        result['weighted_cut_volume_ratio'] = weighted_features['weighted_cut']/unweigthed_features['volume']
        result['surface_area_volume_ratio'] = unweigthed_features['surface_area']/unweigthed_features['volume']
        return result

    @staticmethod
    def cauchy_function(i1, i2, M, beta=10 ^ 4):
        """
        Cauchy distribution function
        :param i1: first image intensity
        :param i2: second image intensity
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :param beta: sensitivity control (default:10^4)
        :return: cauchy distribution value of image intensities
        """
        return 1 / (1 + beta * ((i1 - i2) / M) ** 2)

    @staticmethod
    def cauchy_function_plus(i1, i2, M, beta=10 ^ 4):
        """
        Outgoing cauchy distribution function
        * if I1 <= I2: 1, else: cauchy function
        :param i1: first image intensity
        :param i2: second image intensity
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :param beta: sensitivity control (default:10^4)
        :return: outgoing cauchy distribution value of image intensities
        """
        if i1 > i2:
            return SVMFeatures.cauchy_function(i1, i2, M, beta)
        else:
            return 1

    @staticmethod
    def cauchy_function_minus(i1, i2, M, beta=10 ^ 4):
        """
        Incoming cauchy distribution function
        * if I1 > I2 1, else: cauchy function
        :param i1: first image intensity
        :param i2: second image intensity
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :param beta: sensitivity control (default:10^4)
        :return: incoming cauchy distribution value of image intensities
        """
        if i1 > i2:
            return 1
        else:
            return SVMFeatures.cauchy_function(i1, i2, M, beta)

    @staticmethod
    def weight_edges_4neighborhood(batch, M, i=1, j=1):
        """
        Calculate average weight of all edges in 4-neighborhood of pixel specified at coordinates(i,j)
        :param batch: part of image (3x3, 2x2, 2x3, 3x2 array)
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :param i: x-position of "center" pixel
        :param j: y-position of "center" pixel
        :return: weights of all edges of pixel in question in 4-neighborhood
        """
        s = 0
        v = batch[i, j]
        counter = 0
        for r in range(0, batch.shape[0]):
            for c in range(0, batch.shape[1]):
                if r == i and c == j:
                    continue
                elif r == i or c == j:
                    s = s + SVMFeatures.cauchy_function(v, batch[r, c], M)
                    counter += 1
        # print(counter)
        return s / counter

    @staticmethod
    def calc_weighted_volume(img, segm, M):
        """
        Calculates weighted volume (= sum over the weights of all voxels inside segmentation)
        :param img: list of images
        :param segm: list of segmentation masks
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :return: weighted volume
        """
        weighted_volume = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if segm[i, j]:  # voxel inside segmentation
                    # 3x3 or 2x2 or 2x3 or 3x2 part of batch
                    idx1 = max(0, i - 1)
                    idx2 = min(img.shape[0], i + 2)
                    idx3 = max(0, j - 1)
                    idx4 = min(img.shape[1], j + 2)
                    if i == img.shape[0]:
                        idx1 = i - 2
                    if j == img.shape[1]:
                        idx3 = i - 2
                    b = img[idx1:idx2, idx3:idx4]
                    # sum up edges weights
                    if (i == 0 and j == img.shape[1]) or (i == img.shape[0] and j == 0) or (i == 0 and j == 0) or (
                            i == img.shape[0] and j == img.shape[1]):  # corner
                        weighted_volume += SVMFeatures.weight_edges_4neighborhood(b, M, min(i, 1), min(j, 1))
                    elif i == 0 or j == 0 or i == img.shape[0] or j == img.shape[1]:  # border
                        weighted_volume += SVMFeatures.weight_edges_4neighborhood(b, M, min(i, 1), min(j, 1))
                    else:  # interior
                        weighted_volume += SVMFeatures.weight_edges_4neighborhood(b, M, 1, 1)
        return weighted_volume

    @staticmethod
    def calc_weighted_cut(img, segm, M):
        """
        Calculates weighted cut (=sum over all edge weights along boundary of segmentation)
        :param img: list of images
        :param segm: list of segmentation masks
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :return: weighted cut
        """
        weighted_cut = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if segm[i, j] == 1:  # voxel inside segmentation & on boundary
                    idx1 = max(0, i - 1)
                    idx2 = min(img.shape[0], i + 2)
                    idx3 = max(0, j - 1)
                    idx4 = min(img.shape[1], j + 2)
                    if i == img.shape[0]:
                        idx1 = i - 2
                    if j == img.shape[1]:
                        idx3 = i - 2
                    for r in range(idx1, idx2):
                        for c in range(idx3, idx4):
                            if r != i and j != c:
                                continue
                            elif (r == i or j == c) and segm[r, c] == 0:  # not vertex and not inside segmentation
                                weighted_cut += SVMFeatures.cauchy_function(img[i, j], img[r, c], M)
                                # print(weighted_cut)
        return weighted_cut

    @staticmethod
    def calc_weighted_curvature(img, segm, M, k=1):
        """
        Calculates weighted curvature (=sum over all edge weights * mean curvature along boundary of segmentation)
        :param img: list of images
        :param segm: list of segmentation masks
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :return: weighted curvature
        """
        # mean curvature matrix
        mean_curvature = SVMFeatures._calc_curvature(segm, k)
        # sum and weighted but component
        weighted_cut = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if segm[i, j] == 1:  # voxel inside segmentation & on boundary
                    idx1 = max(0, i - 1)
                    idx2 = min(img.shape[0], i + 2)
                    idx3 = max(0, j - 1)
                    idx4 = min(img.shape[1], j + 2)
                    if i == img.shape[0]:
                        idx1 = i - 2
                    if j == img.shape[1]:
                        idx3 = i - 2
                    for r in range(idx1, idx2):
                        for c in range(idx3, idx4):
                            if r != i and j != c:
                                continue
                            elif (r == i or j == c) and segm[r, c] == 0:  # not vertex and not inside segmentation
                                weighted_cut += SVMFeatures.cauchy_function(img[i, j], img[r, c], M) * mean_curvature[
                                    r, c]
                                # print(weighted_cut)
        return weighted_cut

    @staticmethod
    def calc_low_hi_weighted_cut(img, segm, M):
        """
        Calculates weighted cut of outgoing edges (=sum of all outgoing edge weights along boundary of segmentation)
        :param img: image
        :param segm: segmentation prediction
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :return: weighted cut
        """
        weighted_cut = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if segm[i, j] == 1:  # voxel inside segmentation & on boundary
                    idx1 = max(0, i - 1)
                    idx2 = min(img.shape[0], i + 2)
                    idx3 = max(0, j - 1)
                    idx4 = min(img.shape[1], j + 2)
                    if i == img.shape[0]:
                        idx1 = i - 2
                    if j == img.shape[1]:
                        idx3 = i - 2
                    for r in range(idx1, idx2):
                        for c in range(idx3, idx4):
                            if r != i and j != c:
                                continue
                            elif (r == i or j == c) and segm[r, c] == 0:  # not vertex and not inside segmentation
                                weighted_cut += SVMFeatures.cauchy_function_plus(img[i, j], img[r, c], M)
                                # print(weighted_cut)
        return weighted_cut

    @staticmethod
    def calc_hi_low_weighted_cut(img, segm, M):
        """
        Calculates weighted cut of incoming edges (=sum of all incoming edge weights along boundary of segmentation)
        :param img: image
        :param segm: segmentation prediction
        :param M: scaling factor (maximum L1 norm of all intensity gradients within segm mask)
        :return: weighted cut
        """
        weighted_cut = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if segm[i, j] == 1:  # voxel inside segmentation & on boundary
                    idx1 = max(0, i - 1)
                    idx2 = min(img.shape[0], i + 2)
                    idx3 = max(0, j - 1)
                    idx4 = min(img.shape[1], j + 2)
                    if i == img.shape[0]:
                        idx1 = i - 2
                    if j == img.shape[1]:
                        idx3 = i - 2
                    for r in range(idx1, idx2):
                        for c in range(idx3, idx4):
                            if r != i and j != c:
                                continue
                            elif (r == i or j == c) and segm[r, c] == 0:  # not vertex and not inside segmentation
                                weighted_cut += SVMFeatures.cauchy_function_minus(img[i, j], img[r, c], M)
                                # print(weighted_cut)
        return weighted_cut

    @staticmethod
    def dice_coeff(gts, preds, k=1):
        """
        Calculates dice coefficient (= 2*overlap/sum)
        1 - perfect segmentation, 0 - no overlap
        :param gts: list of groundtruths
        :param preds: list of predicts
        :param k: value of segmentation (default: 1)
        :return: dice coefficient between gt and prediction
        """
        result = []
        for gt, pred in zip(gts, preds):
            if ~(gt == k).any():  # no prediction
                result.append(np.nan)
                continue
            result.append(np.sum(pred[gt == k]) * 2.0 / (np.sum(pred) + np.sum(gt)))
        return result

    @staticmethod
    def jaccard_distance(gts, preds, k=1):
        """
        Calculates jaccard distance (=overlap/union)
        1 - perfect segmentation, 0 - no overlap
        :param gts: list of groundtruths
        :param preds: list of predicts
        :param k: value of segmentation (default: 1)
        :return: list of dice coefficient between gt and prediction
        """
        result = []
        for gt, pred in zip(gts, preds):
            if ~(gt == k).any():  # no prediction
                result.append(np.nan)
                continue
            result.append(np.sum(pred[gt == k]) / (np.sum(pred) + np.sum(gt) - np.sum(pred[gt == k])))
        return result

    @staticmethod
    def hausdorff_distance(gts, preds, k=1):
        """
        Calculates hausdorff distance
        0 - perfect overlap
        the bigger, the larger the distance between surfaces
        :param gts: list of groundtruths
        :param preds: list of predicts
        :param k: value of segmentation (default: 1)
        :return: list of hausdorff distance between gt and prediction
        """
        result = []
        for gt, pred in zip(gts, preds):
            if ~(gt == k).any():  # no prediction
                result.append(np.nan)
                continue
            distance = cdist(gt, pred, 'euclidean')
            dist1 = np.max(np.min(distance, axis=0))
            dist2 = np.max(np.min(distance, axis=1))
            result.append(max(dist1, dist2))
        return result

    @staticmethod
    def mod_hausdorff_distance(gts, preds, k=1):
        """
        Calculates a modified hausdorff distance on surface (instead of max use mean)
        0 - perfect overlap
        the bigger, the larger the distance between surfaces
        :param gts: list of groundtruths
        :param preds: list of predicts
        :param k: value of segmentation (default: 1)
        :return: list of hausdorff distance between gt and prediction
        """
        result = []
        for gt, pred in zip(gts, preds):
            if ~(gt == k).any():  # no prediction
                result.append(np.nan)
                continue
            gt3, contours, hierarchy = cv2.findContours(gt.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            gt_contour = cv2.drawContours(np.zeros_like(gt3), contours, -1, k, 1)
            pred3, contours, hierarchy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            pred_contour = cv2.drawContours(np.zeros_like(pred3), contours, -1, k, 1)
            distance = cdist(gt_contour, pred_contour, 'euclidean')
            dist1 = np.mean(np.min(distance, axis=0))
            dist2 = np.mean(np.min(distance, axis=1))
            result.append(max(dist1, dist2))
        return result

    @staticmethod
    def average_surface_error(gts, preds, k=1):
        """
        Calculates average surface error (=mean per-vertex surface distance)
        :param gts: list of groundtruths
        :param preds: list of predicts
        :param k: value of segmentation (default: 1)
        :return: list of average surface error between gt and prediction
        """
        result = []
        for gt, pred in zip(gts, preds):
            if ~(gt == k).any():  # no prediction
                result.append(np.nan)
                continue
            gt3, contours, hierarchy = cv2.findContours(gt.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            gt_contour = cv2.drawContours(np.zeros_like(gt3), contours, -1, k, 1)
            pred3, contours, hierarchy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            pred_contour = cv2.drawContours(np.zeros_like(pred3), contours, -1, k, 1)
            distance = cdist(gt_contour, pred_contour, 'euclidean')
            dist1 = np.sum(np.min(distance, axis=0)) / np.sum(pred_contour == k)  # first term
            dist2 = np.sum(np.min(distance, axis=1)) / np.sum(gt_contour == k)  # second term
            result.append(1 / 2 * (dist1 + dist2))
        return result
