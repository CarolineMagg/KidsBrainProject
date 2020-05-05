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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class Segmentation():

    def __init__(self, patientData, debug=False):
        """
        Constructor
        """

        self.patient = patientData
        self.debug = debug
        self.result = pd.DataFrame(columns=self.patient.get_filtered_contour_names().values)
        self._reset_stack_tmp()

    def _reset_stack_tmp(self):
        self._stack_img = None
        self._stack_contour_init = None
        self._stack_pts_init = None
        self._stack_pts_dilated = []
        self._stack_pts_segm = []

    @staticmethod
    def pts_to_contour(pts, contour_shape, value=(255, 0, 0)):
        contour = np.zeros(contour_shape, dtype=np.int16)
        vertices = pts.astype(np.int32)
        if len(vertices) != 0:
            cv2.drawContours(contour, [vertices], -1, value, -1)
        return contour

    @staticmethod
    def contour_to_pts(contour):
        pts = []
        tmp = cv2.findContours(contour.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        for t in tmp:
            pts.append(t.reshape(t.shape[0], t.shape[2]))
        return pts

    def _select_data(self, struct, postprocess=0, first=0, last=None):
        self._stack_img = self.patient.get_post_images()[postprocess][first:last]
        self._stack_contour_init = self.patient.get_contour_overlay(struct)[first:last]
        self._stack_pts_init = self.patient.get_contour_points(struct)[first:last]
        self._tmp_struct = struct
        self._tmp_first = first

    def active_contour(self, struct, postprocess=0, first=0, last=None,
                       kernel=(10, 10), w_edge=150, beta=2,
                       debug=False):
        self._reset_stack_tmp()
        if last is None:
            last = first + 1
        log.info(' Select data.')
        self._select_data(struct, postprocess, first, last)
        stack_contour_segm = []
        log.info(' Start segmentation of %s.' % struct)
        for idx, image in enumerate(self._stack_img):
            t = time.time()
            contour_init = self._stack_contour_init[idx]
            pts_init = self._stack_pts_init[idx]
            if pts_init is not None:
                pts_dilated = Segmentation.dilate_segmentation(contour_init, kernel_size=kernel, debug=debug)
                self._stack_pts_dilated.append(pts_dilated)
                contour_proc = []
                pts_proc = []
                for pts_dilate in pts_dilated:
                    pts = segmentation.active_contour(image, pts_dilate, w_edge=w_edge, beta=beta)
                    contour_proc.append(Segmentation.pts_to_contour(pts, image.shape))
                    pts_proc.append(pts)
                self._stack_pts_segm.append(pts_proc)
                stack_contour_segm.append(contour_proc)
            else:
                stack_contour_segm.append(contour_init)
                self._stack_pts_dilated.append(pts_init)
                self._stack_pts_segm.append(pts_init)
            elapsed = time.time() - t
            log.info(' ... slice: %s, time: %s', first + idx, elapsed)
        return stack_contour_segm

    @staticmethod
    def dilate_segmentation(contour_mask_init, kernel_size=(10, 10), iteration=1, debug=False):
        kernel = np.ones(kernel_size, np.uint8)
        contour_dilated = cv2.dilate(contour_mask_init, kernel, iterations=iteration)
        pts_dilated = Segmentation.contour_to_pts(contour_dilated)
        return pts_dilated

    def show_segmentation_single(self, index=0):

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
