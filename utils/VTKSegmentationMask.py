########################################################################################################################
# Class to create segmentation mask ready to be rendered by VTK
# This includes segmentation mask modification in numpy
########################################################################################################################
import sys
import os
import numpy as np
import vtk
import cv2
from vtk.util import numpy_support

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class VTKSegmentationMask:

    def __init__(self, path_list, contour_width=1, contour_color=255, bg_color=(1, 1, 1), fill=True):
        pass
        self.path_list = path_list
        self.number_time_steps = len(path_list)
        pngfiles = []
        for path_mask in path_list:
            pngfiles.append([os.path.join(path_mask, x) for x in os.listdir(path_mask)])
        self.png_files = pngfiles
        self.number_slices = len(pngfiles[0])
        self.contour_width = contour_width
        self.contour_color = contour_color
        self.bg_color = bg_color
        self.fill = fill

        self.np_mask_list = []
        self.np_contour_list = []
        self.np_distance_transform = []
        self.vtk_distance = []
        self.vtk_contour = []

        self.x = None
        self.y = None

    def generate(self):
        self.np_mask_list = []
        self.np_contour_list = []
        self.np_distance_transform = []
        self.vtk_distance = []
        self.vtk_contour = []

        self.generate_np_data()
        self.generate_distance_transform()
        self.generate_vkt_data()
        return self.vtk_distance, self.vtk_contour

    def generate_np_data(self):
        png_reader = vtk.vtkPNGReader()
        for pngfile in self.png_files:
            np_mask_tmp = []
            np_contour_tmp = []
            for i, p in enumerate(pngfile):
                # read pngs
                png_reader.SetFileName(p)
                png_reader.Update()
                img_data = png_reader.GetOutput()
                vtk_data = img_data.GetPointData().GetScalars()
                x, y, z = img_data.GetDimensions()
                if self.x is not None and x != self.x:
                    raise RuntimeError("VTKSegmentationMask: x dimensions in png files do not match.")
                if self.y is not None and y != self.y:
                    raise RuntimeError("VTKSegmentationMask: y dimensions in png files do not match.")
                self.x = x
                self.y = y
                # generate mask
                numpy_data = np.flip(numpy_support.vtk_to_numpy(vtk_data).reshape(x, y))
                np_mask_tmp.append(numpy_data)
                # generate contour
                pts = []
                tmp = cv2.findContours(numpy_data.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
                for t in tmp:
                    pts.append(t.reshape(t.shape[0], t.shape[2]))
                contour = np.zeros(numpy_data.shape, dtype=np.int16)
                for p in pts:
                    vertices = p.astype(np.int32)
                    if len(vertices) != 0:
                        cv2.drawContours(contour, [vertices], -1, (self.contour_color), self.contour_width)
                np_contour_tmp.append(contour)
            self.np_mask_list.append(np_mask_tmp)
            self.np_contour_list.append(np_contour_tmp)

    def generate_distance_transform(self):
        for jdx in range(self.number_time_steps - 1):  # time steps
            np_distance_transform_tmp = []
            for idx in range(self.number_slices):  # per slice
                contour1 = (~self.np_mask_list[jdx][idx] / 255).astype(np.uint8)
                contour2 = (~self.np_mask_list[jdx + 1][idx] / 255).astype(np.uint8)
                dist = cv2.distanceTransform(contour1, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                dist[contour2 != 0] = 0
                if np.max(dist) != 0:
                    dist = dist / np.max(dist) * 255
                if jdx == 0 and self.fill:
                    dist[contour1 == 0] = 1
                np_distance_transform_tmp.append(dist)
            self.np_distance_transform.append(np_distance_transform_tmp)

    def generate_vkt_data(self):
        # distance transform segmentation mask
        for idx in range(self.number_time_steps - 1):  # per time step
            vtk_dt = numpy_support.numpy_to_vtk(np.flip(np.array(self.np_distance_transform[idx]).ravel()))
            vtk_imageData = vtk.vtkImageData()
            vtk_imageData.GetPointData().SetScalars(vtk_dt)
            vtk_imageData.SetDimensions((self.x, self.y, self.number_slices))
            vtk_imageData.SetSpacing(1.0, 1.0, 1.0)
            vtk_imageData.SetOrigin(0.0, 0.0, 0.0)
            self.vtk_distance.append(vtk_imageData)

        # contour masks
        for idx in range(self.number_time_steps):
            # contour masks
            vtk_c = numpy_support.numpy_to_vtk(np.flip(np.array(self.np_contour_list[idx]).ravel()))
            vtk_imageData = vtk.vtkImageData()
            vtk_imageData.GetPointData().SetScalars(vtk_c)
            vtk_imageData.SetDimensions((self.x, self.y, self.number_slices))
            vtk_imageData.SetSpacing(1.0, 1.0, 1.0)
            vtk_imageData.SetOrigin(0.0, 0.0, 0.0)
            self.vtk_contour.append(vtk_imageData)