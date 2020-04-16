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
        pre-treatment dicom is used as reference for all the other dicom files (correction factor)
        :return:
        """

        print("\nPreop dicom")
        self._preop_dicoms = DicomWrapper(self._preop_dicom_path, True, True)
        self._correction = self._preop_dicoms.get_correction()
        for postop_path in self._postop_dicom_path:
            print("\nPostop", postop_path)
            self._postop_dicoms.append(DicomWrapper(postop_path, True, True))

    def get_pre_images(self, ind=None):
        return self._preop_dicoms.get_images(ind)

    def get_post_images(self, ind=None):
        return [x.get_images(ind) for x in self._postop_dicoms]

    def get_slice_location(self, ind=None):
        return self._preop_dicoms.get_slice_location(ind)

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

    def filter_contour_list(self, roiname=None, roinumber=None, mode="approx"):
        """
        Method to filter entire contour list for specific ROI
        :param roiname: list of ROINames to filter
        :param roinumber: list of ROINumbers to filter
        :param mode: either approx (default) - given roiname is part of ROIName
                        or exact - names must exactly match
        :return: subset of pandas dataframe with requested ROI
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
        return self.contour_list_names_filtered

    def read_filtered_contour(self):
        """
        Method to read the contour information of filtered roi names
        :return: dict with roi names as keys and list of layer-wise coordinates
        """

        if len(self.contour_list_names_filtered) == 0:
            raise ValueError(
                "List of contours of interest is empty. Please run filter_contour_list before filter_contour.")

        contours_dict_point_layerwise = {x: [] for x in self.contour_list_names_filtered['RoiName']}
        for idx, x in self.contour_list_names_filtered.iterrows():
            contour_points_layerwise = []
            i = x['ID']
            name = x['RoiName']
            for j in range(self._contour_dcm.ROIContourSequence.__getitem__(i).ContourSequence.__len__()):
                contour = np.array(
                    self._contour_dcm.ROIContourSequence.__getitem__(i).ContourSequence.__getitem__(j).ContourData)
                contour = contour.reshape(int(contour.size / 3.), 3) + self._correction
                contour_points_layerwise.append(contour)
            contours_dict_point_layerwise[name] = contour_points_layerwise

        self.contour_dict_points_layerwise = contours_dict_point_layerwise

    def show_overlay_pre(self, ind, struct):
        """
        Method to show the overlay of pre-treatment image and contour
        :param ind: slice index
        :param struct: name of roi of interest
        :return:
        """
        img_preop = self.get_pre_images(ind)
        contour_img = self.create_contour_overlay(ind, struct)
        overlay_pre = cv2.addWeighted(img_preop.astype(np.uint16), 1.0, contour_img, 0.8, 0)
        plt.imshow(overlay_pre)
        plt.show()

    def show_overlay_post(self, ind, struct):
        """
        Method to show the overlay of post-treatment image and contour
        :param ind: slice index
        :param struct: name of roi of interest
        :return:
        """
        img_postop = self.get_post_images(ind)
        contour_img = self.create_contour_overlay(ind, struct)
        for img in img_postop:
            overlay_post = cv2.addWeighted(img.astype(np.uint16), 1.0, contour_img, 0.8, 0)
            plt.imshow(overlay_post)
            plt.show()

    def get_contour(self, ind, struct):
        """
        Method to get contour coordinates of given structure and slice
        :param ind: slice index
        :param struct: name of roi of interest
        :return:
        """
        if struct not in self.contour_dict_points_layerwise.keys():
            raise ValueError("ROIName was not filtered.")
        loc = self.get_slice_location(ind)
        vertices = np.array([x[:,0:2] for x in self.contour_dict_points_layerwise[struct] if x[0][2] == loc])
        if len(vertices) != 0:
            vertices = vertices[0].astype(np.int32)
        return vertices

    def create_contour_overlay(self, ind, struct):
        """
        Method to create the contour 2D mask of slice ind and given roi name
        :param ind: slice index
        :param struct: name of roi of interest
        :return:
        """
        contour_img = np.zeros_like(self.get_pre_images(ind))
        vertices = self.get_contour(ind, struct)
        if len(vertices) != 0:
            cv2.drawContours(contour_img, [vertices], -1, (255, 255, 255), -1)
        else:
            print('No contour')
        return contour_img

    def show_overlays(self, ind, struct):
        """
        Method to show overlays for pre- and post-treatment data
        :param ind: slice index
        :param struct: name of roi of interest
        :return:
        """
        self.show_overlay_pre(ind, struct)
        self.show_overlay_post(ind, struct)
