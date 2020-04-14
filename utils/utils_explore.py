########################################################################################################################
# File to collect methods to read dicom files
# refactored to be part of DataStructureDicom wrapper for Dicom data
########################################################################################################################
import pandas as pd
import pydicom
import numpy as np
import os
# TODO: add logging to avoid general print commands

__author__ = "c.magg"


def read_structure(path_dicom_folder):
    """
    Method to read structure of folder with dicom files
    :param path_dicom_folder: path to folder with dicom files
    :return: pandas dataframe with relative path, slicelocation and modality type
    """
    df = pd.DataFrame(columns=['Path', 'SliceLocation', 'Type'])
    for dir_name, subdir_list, file_list in os.walk(path_dicom_folder):
        for fn in file_list:
            if ".dcm" in fn.lower():
                dcm = pydicom.dcmread(os.path.join(dir_name, fn))
                row = {'Path': os.path.abspath(os.path.join(dir_name, fn)),
                       'SliceLocation': dcm.SliceLocation, 'Type': dcm.Modality}
                df = df.append(row, ignore_index=True)
    df.sort_values(by=['SliceLocation'])
    return df


def read_contour_names(path_contour):
    """
    Method to read contour names from a dicom file with contour information
    :param path_contour: path to file with dicom contour information
    :return: pandas dataframe with RoiNumber and RoiName and ID in StructureSetRoiSequence
    """
    df = pd.DataFrame(columns=['ID', 'RoiNumber', 'RoiName'])
    if ".dcm" in path_contour.lower():
        dcm = pydicom.dcmread(path_contour)
        print('number of ROIs', str(dcm.StructureSetROISequence.__len__()))
        for i in range(dcm.StructureSetROISequence.__len__()):
            row = {'ID': i,
                   'RoiNumber': dcm.StructureSetROISequence.__getitem__(i).ROINumber,
                   'RoiName': dcm.StructureSetROISequence.__getitem__(i).ROIName}
            df = df.append(row, ignore_index=True)
    return df


def read_contour(path_contour, df_contour):
    """
    Method to read contour information from dicom file
    :param path_contour: path to file with dicom contour information
    :param df_contour: pandas dataframe with information about RoiName
    :return: list of contour coordinates
    """
    list_contour = []
    if ".dcm" in path_contour.lower():
        dcm = pydicom.dcmread(path_contour)
        print('number of ROIs', str(len(df_contour)))
        for i in range(len(df_contour)):
            idx = df_contour.iloc[i]['ID']
            print(idx, df_contour.iloc[i]['RoiName'])
            print('number of contour', str(dcm.ROIContourSequence.__getitem__(idx).ContourSequence.__len__()))
            for j in range(dcm.ROIContourSequence.__getitem__(idx).ContourSequence.__len__()):
                contour = np.array(dcm.ROIContourSequence.__getitem__(idx).ContourSequence.__getitem__(j).ContourData)
                contour = contour.reshape(int(contour.size / 3.), 3)
                list_contour.append(contour)

    print(np.array(list_contour).size, np.array(list_contour).shape)
    return list_contour


def read_contour_row(path_contour, df_contour):
    """
    Method to read contour information from dicom file row-wise
    :param path_contour: path to file with dicom contour information
    :param df_contour: pandas dataframe with information about RoiName
    :return: numpy array of contour coordinates
    """
    list_contour = []
    if ".dcm" in path_contour.lower():
        dcm = pydicom.dcmread(path_contour)
        print('number of ROIs', str(len(df_contour)))
        for i in range(len(df_contour)):
            idx = df_contour.iloc[i]['ID']
            print(idx, df_contour.iloc[i]['RoiName'])
            print('number of contour', str(dcm.ROIContourSequence.__getitem__(idx).ContourSequence.__len__()))
            for j in range(dcm.ROIContourSequence.__getitem__(idx).ContourSequence.__len__()):
                contour = np.array(dcm.ROIContourSequence.__getitem__(idx).ContourSequence.__getitem__(j).ContourData)
                contour = contour.reshape(int(contour.size / 3.), 3)
                for k in range(contour.shape[0]):
                    list_contour.append(contour[k])

    return np.array(list_contour)
