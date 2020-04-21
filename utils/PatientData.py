########################################################################################################################
# Class to structure Dicom data and convenient methods
########################################################################################################################
import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.DicomWrapper import DicomWrapper

__author__ = "c.magg"


class PatientData:

    def __init__(self, pre_dicom_path, post_dicom_path, contours_dicom_path, debug=True):
        """
        Constructor
        :param pre_dicom_path: dicom folder for ct data (pre-treatment)
        :param post_dicom_path: dicom folder for mri data (post-treatment)
        :param contours_dicom_path: dicom file of contour information
        :param debug: boolean for debug information
        """
        self._debug = debug
        self._preop_dicom_path = pre_dicom_path
        self._postop_dicom_path = post_dicom_path
        self._contour_dicom_path = contours_dicom_path
        self._preop_dicoms = None
        self._correction = None
        self._postop_dicoms = []
        self._contour_dcm = None
        self.contour_list_names = pd.DataFrame(columns=['ID', 'RoiNumber', 'RoiName'])
        self.contour_list_names_filtered = pd.DataFrame(columns=['ID', 'RoiNumber', 'RoiName'])
        self.contour_dict_points_layerwise = {}
        self._read_dicom()
        self._read_dicom_contour()
        self._read_contour_list()

    def _read_dicom(self):
        """
        Method to create DicomWrapper objects for pre and post treatment dicoms
        pre-treatment dicom is used as reference for all the other dicom files (correction factors)
        :return:
        """
        print("\nPreop dicom")
        self._preop_dicoms = DicomWrapper(self._preop_dicom_path, True, True)
        self._correction = self._preop_dicoms.get_correction()
        for postop_path in self._postop_dicom_path:
            print("\nPostop", postop_path)
            self._postop_dicoms.append(DicomWrapper(postop_path, True, True))

    def _read_dicom_contour(self):
        """
        Method to read dicom contour file
        :return:
        """
        if ".dcm" not in self._contour_dicom_path.lower():
            raise TypeError("File needs to be dicom format.")
        try:
            self._contour_dcm = pydicom.dcmread(self._contour_dicom_path)
        except:
            raise ImportError("Dicom-file contours couldn't be opened.")

        if self._debug:
            if 'PixelSpacing' in self._contour_dcm:
                print("Pixel spacing....:", self._contour_dcm.PixelSpacing)

    def _read_contour_list(self):
        """
        Method to read contour information of dicom contour file and extract ROINumber and ROIName
        :return:
        """
        if len(self.contour_list_names) != 0:
            raise Warning("Contour list is already initialized and will not be initialized again.")
        for i in range(self._contour_dcm.StructureSetROISequence.__len__()):
            row = {'ID': i,
                   'RoiNumber': self._contour_dcm.StructureSetROISequence.__getitem__(i).ROINumber,
                   'RoiName': self._contour_dcm.StructureSetROISequence.__getitem__(i).ROIName}
            self.contour_list_names = self.contour_list_names.append(row, ignore_index=True)

    def get_pre_images(self, slice_index=None):
        """
        Getter
        :param slice_index: index of slice, if None entire volume
        :return: list of pre-treatment images
        """
        return self._preop_dicoms.get_images(slice_index)

    def get_post_images(self, slice_index=None):
        """
        Getter
        :param slice_index: index of slice, if None entire volume
        :return: list of post-treatment images for all post-treatment dicoms
        """
        if slice_index is None:
            return [x.get_images(slice_index) for x in self._postop_dicoms]
        else:
            return [x.get_images(slice_index)[0] for x in self._postop_dicoms]

    def get_slice_location(self, slice_index=None):
        """
        Getter
        :param slice_index: index of slice, if None entire volume
        :return: location of slice(s)
        """
        return self._preop_dicoms.get_slice_location(slice_index)

    def _filter_contour_list(self, roiname=None, roinumber=None, mode="approx"):
        """
        Method to filter entire contour list for specific ROI(s)
        :param roiname: list of ROINames to filter
        :param roinumber: list of ROINumbers to filter
        :param mode: either approx (default) - given roiname is part of ROIName
                        or exact - names must exactly match
        :return:
        """

        if type(roiname) == str:
            roiname = [roiname]
        if roiname is not None:
            if mode == 'approx':
                self.contour_list_names_filtered = pd.DataFrame(
                    [x for y in roiname for idx, x in self.contour_list_names.iterrows() if y in x['RoiName']])
            if mode == 'exact':
                self.contour_list_names_filtered = pd.DataFrame(
                    [x for y in roiname for idx, x in self.contour_list_names.iterrows() if y == x['RoiName']])

        if roinumber is not None:
            self.contour_list_names_filtered = self.contour_list_names_filtered.append(pd.DataFrame(
                [x for y in roinumber for idx, x in self.contour_list_names.iterrows() if y == x['RoiNumber']]))

        if len(self.contour_list_names_filtered) != 0:
            self.contour_list_names_filtered = self.contour_list_names_filtered.drop_duplicates()

    def read_filtered_contour(self, roiname=None, roinumber=None, mode="approx"):
        """
        Method to read the contour information of roi names/numbers of interest
        :param roiname: list of ROINames to filter
        :param roinumber: list of ROINumbers to filter
        :param mode: either approx (default) - given roiname is part of ROIName
                        or exact - names must exactly match
        :return: subset of pandas dataframe with unique requested ROI
        """

        self._filter_contour_list(roiname=roiname, roinumber=roinumber, mode=mode)

        if len(self.contour_list_names_filtered) == 0:
            raise ValueError(
                "List of contours of interest is empty. Please set either roiname or roinumber.")

        contours_dict_point_layerwise = {x: [] for x in self.contour_list_names_filtered['RoiName']}
        for idx, x in self.contour_list_names_filtered.iterrows():
            contour_points_layerwise = []
            i = x['ID']
            name = x['RoiName']
            for j in range(self._contour_dcm.ROIContourSequence.__getitem__(i).ContourSequence.__len__()):
                contour = np.array(
                    self._contour_dcm.ROIContourSequence.__getitem__(i).ContourSequence.__getitem__(j).ContourData)
                contour = contour.reshape(int(contour.size / 3.),
                                          3) + self._correction  # correction to get positive values
                contour_points_layerwise.append(contour)
            contours_dict_point_layerwise[name] = contour_points_layerwise
        self.contour_dict_points_layerwise = contours_dict_point_layerwise
        return self.contour_list_names_filtered

    def get_contour_points(self, struct, slice_index=None):
        """
        Method to get contour coordinates of structure in slice(s)
        :param struct: name of roi of interest
        :param slice_index: index of slice(s), if None entire volume
        :return: list of vertices per slice
        """
        if struct not in self.contour_dict_points_layerwise.keys():
            raise ValueError("ROIName was not filtered.")
        if slice_index is None:
            vertices = [x[:, 0:2].astype(np.int32) for x in self.contour_dict_points_layerwise[struct]]
        else:
            loc = self.get_slice_location(slice_index)
            vertices = [x[:, 0:2].astype(np.int32) for x in self.contour_dict_points_layerwise[struct] if x[0][2] == loc]
        return vertices

    def create_contour_overlay(self, struct, slice_index=None):
        """
        Method to create the contour 2D mask of slice index and given roi name
        :param struct: name of roi of interest
        :param slice_index: index of slice(s), if None entire volume
        :return: contour 2D mask as np.uint16 array with 0 - background, 255 - mask
        """
        ct_shape = [x.shape for x in self.get_pre_images(slice_index)]
        contour_shape = [
            (int(np.round(self._preop_dicoms.get_pixel_spacing()[0] * x[0])),
             int(np.round(self._preop_dicoms.get_pixel_spacing()[1] * x[1]))) for
            x in ct_shape]
        contour_img = [np.zeros(x, dtype=np.uint16) for x in contour_shape]
        vertices = self.get_contour_points(struct, slice_index)
        idx_last = 0
        if slice_index is None:
            last_loc = self.contour_dict_points_layerwise[struct][-1][0][2]
            idx_last = np.where(self._preop_dicoms.get_slice_location() == last_loc)[0][0]
        [cv2.drawContours(contour_img[idx + idx_last], [x], -1, (255, 255, 255), -1) for idx, x in
         enumerate(reversed(vertices))]
        contour_img_res = [cv2.resize(x, dsize=ct_shape[slice_index], interpolation=cv2.INTER_NEAREST) for slice_index, x in
                           enumerate(contour_img)]
        return contour_img_res

    def show_overlay2D_pre(self, struct, slice_index):
        """
        Method to show the overlay of pre-treatment image and contour for single slice
        :param struct: name of roi of interest
        :param slice_index: index of slice
        :return:
        """
        img_preop = self.get_pre_images(slice_index)[0]
        contour_img = self.create_contour_overlay(struct, slice_index)[0]
        overlay_pre = cv2.addWeighted(img_preop, 1.0, contour_img, 0.8, 0)
        plt.title('slice %d' % slice_index)
        plt.imshow(overlay_pre)
        plt.axis('off')
        plt.show()

    def show_overlay2D_post_init(self, struct, slice_index):
        """
        Method to show the overlay of all post-treatment images and initial contour for single slice
        :param struct: name of roi of interest
        :param slice_index: index of slice
        :return:
        """
        img_postop = self.get_post_images(slice_index)
        contour_img = self.create_contour_overlay(struct, slice_index)[0]
        fig, ax = plt.subplots(1, len(img_postop), figsize=[12, 12])
        for idx, img in enumerate(img_postop):
            overlay_post = cv2.addWeighted(img.astype(np.uint16), 1.0, contour_img, 0.6, 0)
            ax[idx].set_title('slice %d' % slice_index)
            ax[idx].imshow(overlay_post)
            ax[idx].axis('off')
        plt.show()

    def show_overlays_init(self, struct, slice_index):
        """
        Method to show overlays for pre- and post-treatment data for single slice
        :param struct: name of roi of interest
        :param slice_index: index of slice
        :return:
        """
        self.show_overlay2D_pre(struct, slice_index)
        self.show_overlay2D_post_init(struct, slice_index)

    # adapted from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
    @staticmethod
    def show_slices2D(stack, rows=6, cols=6, start_with=10, show_every=3):
        """
        Method to show slices of entire volume in 2D
        :param stack: list of images (stack of images)
        :param rows: number of rows
        :param cols: number of cols
        :param start_with: starting index of stack
        :param show_every: indices which will be shown
        :return:
        """
        fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
        for i in range(rows * cols):
            slice_index = start_with + i * show_every
            ax[int(i / rows), int(i % rows)].set_title('slice %d' % slice_index)
            ax[int(i / rows), int(i % rows)].imshow(stack[slice_index], cmap='gray')
            ax[int(i / rows), int(i % rows)].axis('off')
        plt.show()

    def show_slices2D_contour(self, struct, rows=6, cols=6, start_with=10, show_every=3):
        """
        Method to show slices of entire pre-treatment volume with contours in 2D
        :param struct: name of roi of interest
        :param rows: number of rows
        :param cols: number of cols
        :param start_with: starting index of stack
        :param show_every: indices which will be shown
        :return:
        """
        img = self.get_pre_images()
        contour_init = self.create_contour_overlay(struct)
        toshow = [cv2.addWeighted(img[idx], 1.0, x, 0.6, 0) for idx, x in enumerate(contour_init)]
        self.show_slices2D(toshow, rows, cols, start_with, show_every)
