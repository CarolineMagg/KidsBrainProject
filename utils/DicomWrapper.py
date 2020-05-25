########################################################################################################################
# Class to wrap Dicom Data into customized data type
########################################################################################################################
import os
import pydicom
import cv2
import numpy as np
from natsort import natsorted
import logging as log

__author__ = "c.magg"


class DicomWrapper:

    def __init__(self, dicom_path, frame=True, debug=True):
        """
        Constructor
        :param dicom_path: path to dicom folder
        :param frame: boolean for switching information about dicom frame on or off
        :param debug: boolean for switching debug information on or off
        """
        self._debug = debug
        self._frame = frame
        self._path_dicom = dicom_path

        self._imgs = None
        self._ref_dcm = None
        self._dimensions = None
        self._pixel_spacing = None
        self._position = None
        self._correction = None
        self._slice_location = None
        self._modality = None

        self._read_dicom()
        if self._frame:
            self._read_reference_frame_info()

    def get_correction(self):
        """
        Getter
        :return: correction factor (translation)
        """
        if self._correction is not None:
            return self._correction
        else:
            raise ValueError("Correction is not set.")

    def get_images(self, slice_index=None):
        """
        Getter
        :param slice_index: index of slice, if None entire volume
        :return: list of normalized images as np.uint6 array
        """
        if slice_index is None:
            return [cv2.normalize(x, x, 0, 255, 32).astype(np.uint16) for x in self._imgs]
        else:
            x = self._imgs[slice_index]
            return [cv2.normalize(x, x, 0, 255, 32).astype(np.uint16)]

    def get_slice_location(self, slice_index=None):
        """
        Getter
        :param slice_index: index of slice, if None entire volume
        :return: location of slice(s)
        """
        if self._slice_location is not None:
            if slice_index is None:
                return self._slice_location
            else:
                return self._slice_location[slice_index]
        else:
            raise ValueError("Slice location is not set.")

    def get_pixel_spacing(self):
        """
        Getter
        :return: pixel spacing of volume
        """
        if self._pixel_spacing is not None:
            return self._pixel_spacing
        else:
            raise ValueError("Pixel spacing is not set.")

    def _read_dicom(self):
        """
        Method to read dicom and convert to list of numpy arrays
        :return:
        """
        files = natsorted(os.listdir(self._path_dicom))
        try:
            dcms_list = [pydicom.read_file(os.path.join(self._path_dicom, fn)) for fn in files]
            self._imgs = [dcm.pixel_array for dcm in dcms_list]
            self._slice_location = [float(x.get('SliceLocation', '-100')) for x in dcms_list]
            self._ref_dcm = dcms_list[0]
        except:
            raise ImportError("Dicom-file couldn't be opened.")

    def _read_reference_frame_info(self):
        """
        Method to read reference frame information and determine:
        * dicom dimensions, eg. 512x512x200
        * pixel spacing, eg. 0.978,0.978,1.0
        * position, eg. -250,-250,388
        * translation correction (mapping from reference coordinate sytem to input pixel space), eg. -250,-217,0
        * modality, eg. CT or MRI
        :return:
        """

        try:
            self._dimensions = (int(self._ref_dcm.Rows), int(self._ref_dcm.Columns), len(self._imgs))
            self._pixel_spacing = (float(self._ref_dcm.PixelSpacing[0]), float(self._ref_dcm.PixelSpacing[1]),
                                   float(self._ref_dcm.SliceThickness))
            self._position = tuple((float(p) for p in self._ref_dcm.ImagePositionPatient))
            self._correction = [abs(0 + self._position[0]) + 1, abs(0 + self._position[1]) + 1, 0]
            self._modality = self._ref_dcm.Modality
        except:
            raise ValueError("Reference frame information could not be processed.")

        if self._debug:
            log.debug("Debug on")
            log.debug("Image size.......: %s", self._dimensions)
            log.debug("Pixel spacing....: %s", self._pixel_spacing)
            log.debug("Slices number....: %s", len(self._slice_location))
            log.debug("Slices location...: %s - %s", self._slice_location[0], self._slice_location[-1])
            log.debug("Position.........: %s", self._position)
            log.debug("Resulting correction: %s", self._correction)
