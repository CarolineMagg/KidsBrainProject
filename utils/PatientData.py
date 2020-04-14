########################################################################################################################
# Class to structure Dicom data and convenient methods
########################################################################################################################
import pandas as pd
import pydicom
import numpy as np
import os

__author__ = "c.magg"


class PatientData:

    def __init__(self, preop_dicom_path, contours_dicom_path, debug=True):
        self._debug = debug
        self._preop_dicom_path = preop_dicom_path
        self._contour_dicom_path = contours_dicom_path
        self._preop_imgs = None
        self._preop_imgs_location = None
        self._contour_dcm = None
        self.contour_list_names = pd.DataFrame(columns=['ID', 'RoiNumber', 'RoiName'])
        self.contour_list_names_filtered = pd.DataFrame(columns=['ID', 'RoiNumber', 'RoiName'])
        self.contour_dict_points_layerwise = {}
        self._read_dicom_preop()
        self._read_dicom_contour()
        self._read_contour_list()

    def _read_dicom_preop(self):
        """
        Method to read dicom preop (CT) data and pixel array of dicoms
        :return:
        """
        files = os.listdir(self._preop_dicom_path)
        try:
            preop_dcms_list = [pydicom.read_file(os.path.join(self._preop_dicom_path, fn)) for fn in files]
            self._preop_imgs = [dcm.pixel_array for dcm in preop_dcms_list]
            self._preop_imgs_location = [float(x.get('SliceLocation', '-100')) for x in preop_dcms_list]
        except:
            raise ImportError("Dicom-file preop couldn't be opened.")
        if self._debug:
            for i in range(len(self._preop_imgs_location)):
                print("Slice location...:", self._preop_imgs_location[i])
            ds = preop_dcms_list[0]
            print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
                rows=int(ds.Rows), cols=int(ds.Columns), size=len(ds.PixelData)))
            if 'PixelSpacing' in ds:
                print("Pixel spacing....:", ds.PixelSpacing)
            print("Slices number....:", len(self._preop_imgs_location))

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
        if mode == 'approx':
            self.contour_list_names_filtered = pd.DataFrame(
                [x for y in roiname for idx, x in self.contour_list_names.iterrows() if y in x['RoiName']])
        if mode == 'exact':
            self.contour_list_names_filtered = pd.DataFrame(
                [x for y in roiname for idx, x in self.contour_list_names.iterrows() if y == x['RoiName']])

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
                contour = contour.reshape(int(contour.size / 3.), 3)
                contour_points_layerwise.append(contour)
            contours_dict_point_layerwise[name] = contour_points_layerwise

        self.contour_dict_points_layerwise = contours_dict_point_layerwise
        return self.contour_dict_points_layerwise
