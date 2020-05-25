########################################################################################################################
# Class to wrap segmentation processing of images
########################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
import sys
import logging as log
import time
import skimage.segmentation as segmentation
from scipy.spatial.distance import cdist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class Segmentation:

    def __init__(self, patientData, debug=False):
        """
        Constructor

        """

        self.patient = patientData
        self.debug = debug
        self.result = pd.DataFrame(columns=self.patient.get_filtered_contour_names().values)
        self._reset_stack_tmp()

    def _reset_stack_tmp(self):
        """
        Method to reset temp stack variables
        :return:
        """
        self._stack_img = None
        self._stack_contour_init = None
        self._stack_pts_init = None
        self._stack_pts_dilated = []
        self._stack_pts_segm = []

    @staticmethod
    def pts_to_contour(pts, contour_shape, value=(255, 0, 0)):
        """
        Method to convert point data to mask data
        :param pts: point data
        :param contour_shape: size of segmentation mask
        :param value: values of segmentation mask
        :return: filled segmentation mask
        """
        contour = np.zeros(contour_shape, dtype=np.int16)
        vertices = pts.astype(np.int32)
        if len(vertices) != 0:
            cv2.drawContours(contour, [vertices], -1, value, -1)
        return contour

    @staticmethod
    def contour_to_pts(contour):
        """
        Method to convert segmentation mask to point data
        :param contour: segmentation mask
        :return: point data of contour
        """
        pts = []
        tmp = cv2.findContours(contour.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        for t in tmp:
            pts.append(t.reshape(t.shape[0], t.shape[2]))
        return pts

    def _select_data(self, struct, postprocess=0, first=0, last=None):
        """
        Method to select the stack data
        :param struct: name of structure in question
        :param postprocess: index of post-treatment data
        :param first: first index for stack data
        :param last: last index for stack data
        :return:
        """
        if postprocess == -1:
            self._stack_img = self.patient.get_pre_images()[first:last]
        else:
            self._stack_img = self.patient.get_post_images()[postprocess][first:last]
        self._stack_contour_init = self.patient.get_contour_overlay(struct)[first:last]
        self._stack_pts_init = self.patient.get_contour_points(struct)[first:last]
        self._stack_contour_pred = []
        self._tmp_struct = struct
        self._tmp_first = first

    def active_contour(self, struct, postprocess=0, first=0, last=None,
                       kernel=(10, 10),
                       w_line=0, w_edge=1,
                       alpha=0.1, beta=0.1, gamma=0.01,
                       max_px_move=1.0, max_iterations=2500, convergence=0.1,
                       boundary_condition='periodic'):
        """
        Method to wrap active contour segmentation algorithm
        :param struct: name of structure in question
        :param postprocess: index of post-treatment data
        :param first: first index for stack data
        :param last:  last index for stack data
        :param kernel: kernel size for dilation
        :param w_line: controls attraction to brightness
        :param w_edge: Controls attraction to edges
        :param alpha: snake length shape parameter (higher values makes snake contract faster)
        :param beta: snake smoothness shape parameter (higher values makes snake smoother)
        :param gamma: time stepping parameter
        :param max_px_move: maximum pixel distance to move per iteration
        :param max_iterations: maximum iterations to optimize snake shape
        :param convergence: convergence criteria
        :param boundary_condition: boundary conditions for the contour
        :return:
        """
        self._reset_stack_tmp()
        if last is None:
            last = first + 1
        log.debug(' Select data.')
        self._select_data(struct, postprocess, first, last)
        log.info(' Start segmentation of %s.' % struct)
        for idx, image in enumerate(self._stack_img):
            t = time.time()
            contour_init = self._stack_contour_init[idx]
            pts_init = self._stack_pts_init[idx]
            if pts_init is not None:
                pts_dilated = Segmentation.dilate_segmentation(contour_init, kernel_size=kernel)
                self._stack_pts_dilated.append(pts_dilated)
                contour_proc = []
                pts_proc = []
                for pts_dilate in pts_dilated:
                    pts = segmentation.active_contour(image, pts_dilate, alpha=alpha, beta=beta, gamma=gamma,
                                                      w_line=w_line, w_edge=w_edge,
                                                      max_px_move=max_px_move, max_iterations=max_iterations,
                                                      convergence=convergence, boundary_condition=boundary_condition)
                    contour_proc.append(Segmentation.pts_to_contour(pts, image.shape))
                    pts_proc.append(pts)
                contour_merged = np.zeros_like(image)
                for contour in contour_proc:
                    contour_merged[contour == 255] = 255
                self._stack_pts_segm.append(pts_proc)
                self._stack_contour_pred.append(contour_merged)
            else:
                self._stack_contour_pred.append(contour_init)
                self._stack_pts_dilated.append(pts_init)
                self._stack_pts_segm.append(pts_init)
            elapsed = time.time() - t
            log.debug(' ... slice: %s, time: %s', first + idx, elapsed)
        return self._stack_contour_pred

    @staticmethod
    def dice_coefficient(gt, pred, k=255):
        """
        Method to calculate dice coefficient between predicted and groundtruth segmentation
        :param gt: groundtruth
        :param pred: prediction
        :param k: value of segmentation (default: 255)
        :return: dice coefficient value between 0 and 1
        """
        return np.sum(pred[gt == k]) * 2.0 / (np.sum(pred) + np.sum(gt))

    @staticmethod
    def volumetric_overlap_error(gt, pred, k=255):
        """
        Method to calculate volumetric overlap error between predicted and groundtruth segmentation
        :param gt: groundtruth
        :param pred: prediction
        :param k: value of segmentation (default: 255)
        :return: volumetric overlap error value between 0 and 1
        """
        return 1 - np.sum(pred[gt == k]) * 2.0 / (np.sum(pred + gt))

    @staticmethod
    def mod_hausdorff_distance(gt, pred):
        """
        Method to calculate modified hausdorff distance between predcited and groundtruth segmentation
        :param gt: groundtruth
        :param pred: prediciton
        :return: modified hausdorff distance
        """
        distance = cdist(gt, pred, 'euclidean')
        dist1 = np.mean(np.min(distance, axis=0))
        dist2 = np.mean(np.min(distance, axis=1))
        return max(dist1, dist2)

    def evaluate_segmentation(self, k=255):
        """
        Method to evaluate the stack segmentation
        :return: dice coefficient, volumetric overlap error, modified hausdorff distance
        """
        log.info(' Start evaluation')
        assert (len(self._stack_contour_init) == len(self._stack_contour_pred),
                "Initial segmentation has not the same length as the predicted segmentation.")
        dice = []
        hausdorff = []
        vol_overlap = []
        index = self._tmp_first
        for gt, pred in zip(self._stack_contour_init, self._stack_contour_pred):
            dice.append(self.dice_coefficient(gt, pred, k))
            hausdorff.append(self.mod_hausdorff_distance(gt, pred))
            vol_overlap.append(self.volumetric_overlap_error(gt, pred, k))
            log.debug("... Segmentation error metrics for slice %s \n"
                      "    Dice coefficient: %s \n"
                      "    Volumetric overlap error: %s \n"
                      "    Mod hausdorff distance: %s", index, dice[-1], vol_overlap[-1], hausdorff[-1])
            index += 1
        return dice, vol_overlap, hausdorff

    @staticmethod
    def dilate_segmentation(contour_mask_init, kernel_size=(10, 10), iteration=1):
        """
        Method to build dilated version of input segmentation mask
        :param contour_mask_init: initial segmentation mask
        :param kernel_size: kernel size for dilation
        :param iteration: number of iterations
        :return: dilated point data
        """
        kernel = np.ones(kernel_size, np.uint8)
        contour_dilated = cv2.dilate(contour_mask_init, kernel, iterations=iteration)
        pts_dilated = Segmentation.contour_to_pts(contour_dilated)
        return pts_dilated

    def show_segmentation_single(self, index=0):
        """
        Method to show a single segmentation process with initial, dilated and predicted segmentation contour
        :param index: stack index of image
        :return:
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        if self._stack_img is None or len(self._stack_img) <= index:
            raise ValueError('No segmentation available.')
        ax.imshow(self._stack_img[index], cmap=plt.cm.gray)
        if len(self._stack_pts_segm) >= index:
            for pts_dilated in self._stack_pts_dilated[index]:
                ax.plot(pts_dilated[:, 0], pts_dilated[:, 1], '-g', lw=2,
                        label='dilated')
            for pts_segm in self._stack_pts_segm[index]:
                ax.plot(pts_segm[:, 0], pts_segm[:, 1], '-b', lw=3, label='processed')
        if len(self._stack_pts_init) >= index:
            for pts_init in self._stack_pts_init[index]:
                ax.plot(pts_init[:, 0], pts_init[:, 1], '--r', lw=2, label='init')
        ind = index + self._tmp_first
        ax.set_title('Post-treatment slice %d' % ind)
        ax.legend()
        ax.axis('off')
        plt.show()

    def show_segmentation_stack(self, rows=6, cols=6, start_with=0, show_every=1):
        """
        Method to show a stack information of segmentation process
        with initial, dilated and predicted segmentation contour
        :param rows: number of rows
        :param cols: number of columns
        :param start_with: starting index
        :param show_every: number of which images should be shown
        :return:
        """
        fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
        for i in range(rows * cols):
            slice_index = start_with + i * show_every
            if slice_index >= len(self._stack_img):
                ax[int(i / cols), int(i % cols)].set_title('No slice')
                ax[int(i / cols), int(i % cols)].imshow(np.zeros_like(self._stack_img[-1]), cmap='gray')
            else:
                ax[int(i / cols), int(i % cols)].set_title('slice %d' % slice_index)
                ax[int(i / cols), int(i % cols)].imshow(self._stack_img[slice_index], cmap=plt.cm.gray)
                if len(self._stack_pts_segm) >= slice_index:
                    for pts_dilated in self._stack_pts_dilated[slice_index]:
                        ax[int(i / cols), int(i % cols)].plot(pts_dilated[:, 0],
                                                              pts_dilated[:, 1],
                                                              '-g', lw=2, label='dilated')
                    for pts_segm in self._stack_pts_segm[slice_index]:
                        ax[int(i / cols), int(i % cols)].plot(pts_segm[:, 0],
                                                              pts_segm[:, 1], '-b',
                                                              lw=3, label='processed')
                if len(self._stack_pts_segm) >= slice_index:
                    for pts_init in self._stack_pts_init[slice_index]:
                        ax[int(i / cols), int(i % cols)].plot(pts_init[:, 0],
                                                              pts_init[:, 1], '--r',
                                                              lw=2, label='init')
            ax[int(i / cols), int(i % cols)].axis('off')
        plt.show()
