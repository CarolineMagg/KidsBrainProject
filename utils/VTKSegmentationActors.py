########################################################################################################################
# Class to create segmentation mask ready to be rendered by VTK
# This includes segmentation mask modification in numpy
########################################################################################################################
import logging
import sys
import os
import vtk

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class VTKSegmentationActors:

    def __init__(self, vtk_mask=None, vtk_contour=None, slice_number=0, bg_color=(1, 1, 1), color_map='rgbo'):
        self.ctf = []
        self.otf = []
        self.lut = []
        self.lut_contour = []
        self.bg_color = bg_color
        self.slice_number = slice_number
        if vtk_mask is None or vtk_contour is None:
            self.vtk_contour = None
            self.vtk_mask = None
            self.number_time_steps = None
        else:
            self.set_data(vtk_mask, vtk_contour)
        self.actors_mask = None
        self.actors_contour = []
        self.pred = None
        self.accuracy = [0.9, 0.9, 0.9, 0.9]

        self.hue = []
        self.value = []
        self.contour_colors = []
        self.define_colormap(color_map)

    def define_colormap(self, color_map):
        """
        Maps color map name to values for hue & value of distance mask and contour colors
        :param color_map:
        :return:
        """
        self.hue = []
        self.value = []
        self.contour_colors = []
        if color_map == 'rgbo':
            self.hue = [0, 240 / 360, 120 / 360, 30 / 360]  # rgbo
            self.value = [0.9, 0.9, 0.6, 0.9]
            self.contour_colors = [(230 / 255, 0, 0),
                                   (0, 0, 230 / 255),
                                   (0, 153 / 255, 0),
                                   (255 / 255, 127 / 255, 0)]
        elif color_map == 'plasma':
            self.hue = [242 / 360, 297 / 360, 13 / 360, 62 / 360]
            self.value = [0.5, 0.6, 0.9, 0.97]
            self.contour_colors = [(13 / 255, 8 / 255, 135 / 255),
                                   (154 / 255, 22 / 255, 159 / 255),
                                   (236 / 255, 119 / 255, 84 / 255),
                                   (240 / 255, 249 / 255, 33 / 255)]
        elif color_map == 'viridis':
            self.hue = [288 / 360, 205 / 360, 151 / 360, 53 / 360]
            self.value = [0.32, 0.55, 0.71, 0.99]
            self.contour_colors = [(68 / 255, 1 / 255, 84 / 255),
                                   (49 / 255, 103 / 255, 142 / 255),
                                   (50 / 255, 182 / 255, 122 / 255),
                                   (251 / 255, 231 / 255, 35 / 255)]
        else:
            raise ValueError("VTKSegmentationActor: Colormap {0} not supported.".format(color_map))

    def set_data(self, vtk_mask, vtk_contour):
        assert len(vtk_mask) + 1 == len(vtk_contour)
        self.number_time_steps = len(vtk_contour)
        self.vtk_mask = vtk_mask
        self.vtk_contour = vtk_contour

    def UpdateColorMap(self, color_map):
        self.define_colormap(color_map)

    def UpdateActors(self, vtk_mask=None, vtk_contour=None, slice_number=None, pred=None):
        if slice_number is not None:
            self.slice_number = slice_number
        self.pred = pred
        if vtk_contour is not None and vtk_mask is not None:
            self.set_data(vtk_mask, vtk_contour)
        self.generate_lut_mask()
        self.generate_lut_contour()
        self.generate_actors_contour()
        self.generate_actors_mask()
        return self.actors_mask, self.actors_contour

    def generate_lut_mask(self):
        """
        Generate look up table for mask (distance transform)
        :return:
        """
        self.convert_pred_to_accuracy()
        saturation = [1] + self.accuracy[1:]  # [0.9, 0.9, 0.9, 0.9]
        opacities = [1, 1, 1, 1]
        for idx in range(self.number_time_steps - 1):
            ctf01 = vtk.vtkColorTransferFunction()
            ctf01.AddHSVPoint(0, 0, 0, 0)
            ctf01.AddHSVPoint(1, self.hue[idx], saturation[idx], self.value[idx])
            ctf01.AddHSVPoint(255, self.hue[idx + 1], saturation[idx + 1], self.value[idx + 1])
            self.ctf.append(ctf01)

            # Piecewise Function for opacity
            otf01 = vtk.vtkPiecewiseFunction()
            otf01.AddPoint(0, 0)  # invisible
            otf01.AddPoint(1, opacities[idx])  # first time step accuracy
            otf01.AddPoint(255, opacities[idx + 1])  # next time step accuracy
            self.otf.append(otf01)

        # look up table
        for idx in range(self.number_time_steps - 1):
            tableSize = 255
            lut01 = vtk.vtkLookupTable()
            lut01.SetNumberOfTableValues(tableSize)
            lut01.SetRampToLinear()
            for ii, ss in enumerate([float(xx) for xx in range(tableSize)]):
                cc = self.ctf[idx].GetColor(ss)
                oo = self.otf[idx].GetValue(ss)
                lut01.SetTableValue(ii, cc[0], cc[1], cc[2], oo)
                lut01.Modified()
            lut01.SetRange(0, 255)
            lut01.Build()
            self.lut.append(lut01)

    def generate_lut_contour(self):
        """
        Generate look up table for contour
        :return:
        """
        for idx in range(self.number_time_steps):
            lut_contour = vtk.vtkLookupTable()
            lut_contour.SetNumberOfColors(2)
            lut_contour.SetTableValue(0, 1, 1, 1, 0)
            lut_contour.SetTableValue(1, *self.contour_colors[idx], 1)
            lut_contour.Build()
            self.lut_contour.append(lut_contour)

    def convert_pred_to_accuracy(self):
        self.accuracy = [1, 1, 1, 1]
        if self.pred is not None:
            for idx, p in enumerate(self.pred):
                if p <= 0.5:
                    self.accuracy[idx] = 0.2
                elif p <= 0.7:
                    self.accuracy[idx] = 0.5
                elif p <= 0.9:
                    self.accuracy[idx] = 0.7
                else:
                    self.accuracy[idx] = 1
        logging.debug("VTKSegmentationActors: contour accuracy is mapped to {0}".format(self.accuracy))

    def generate_actors_mask(self):
        self.actors_mask = []
        res = []
        for idx in range(self.number_time_steps - 1):
            scalarValuesToColors = vtk.vtkImageMapToColors()
            scalarValuesToColors.SetLookupTable(self.lut[idx])
            scalarValuesToColors.SetInputData(self.vtk_mask[idx])
            res.append(scalarValuesToColors)

        blend = vtk.vtkImageBlend()
        blend.SetOpacity(0, 0.5)  # 0=1st image, 0.5 alpha of 1st image
        blend.SetOpacity(1, 0.5)  # 1=2nd image, 0.5 alpha of 2nd image
        blend.SetBlendModeToCompound()  # images compounded together and each component is scaled by the sum of the alpha/opacity values
        for idx in range(len(res)):
            blend.AddInputConnection(res[idx].GetOutputPort())

        actors_mask = vtk.vtkImageSlice()
        actors_mask.SetMapper(vtk.vtkImageSliceMapper())
        actors_mask.GetMapper().SetSliceNumber(self.slice_number)
        actors_mask.GetMapper().SetInputConnection(blend.GetOutputPort())
        actors_mask.GetProperty().SetOpacity(1)
        actors_mask.SetPosition(0, 1, 0)
        self.actors_mask = actors_mask

    def generate_actors_contour(self):
        self.actors_contour = []
        for idx in range(self.number_time_steps):
            scalarValuesToColors = vtk.vtkImageMapToColors()
            scalarValuesToColors.SetLookupTable(self.lut_contour[idx])
            scalarValuesToColors.SetInputData(self.vtk_contour[idx])
            scalarValuesToColors.GetOutput()

            # Create an image actor
            actor = vtk.vtkImageSlice()
            actor.SetMapper(vtk.vtkImageSliceMapper())
            actor.GetMapper().SetSliceNumber(self.slice_number)
            actor.GetMapper().SetInputConnection(scalarValuesToColors.GetOutputPort())
            actor.GetProperty().SetOpacity(1)
            self.actors_contour.append(actor)
