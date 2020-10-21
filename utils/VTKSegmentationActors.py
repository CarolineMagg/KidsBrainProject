########################################################################################################################
# Class to create segmentation mask ready to be rendered by VTK
# This includes segmentation mask modification in numpy
########################################################################################################################
import sys
import os
import vtk

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class VTKSegmentationActors:

    def __init__(self, vtk_mask=None, vtk_contour=None, slice_number=0, bg_color=(1, 1, 1)):
        self.ctf = []
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
        self.actors = []
        self.actors_contour = []

    def set_data(self, vtk_mask, vtk_contour):
        assert len(vtk_mask) + 1 == len(vtk_contour)
        self.number_time_steps = len(vtk_contour)
        self.vtk_mask = vtk_mask
        self.vtk_contour = vtk_contour

    def UpdateActors(self, vtk_mask=None, vtk_contour=None, slice_number=None):
        if slice_number is not None:
            self.slice_number = slice_number
        if vtk_contour is not None and vtk_mask is not None:
            self.set_data(vtk_mask, vtk_contour)
        self.generate_lut_mask()
        self.generate_lut_contour()
        self.generate_actors_contour()
        self.generate_actors_mask()
        return self.actors, self.actors_contour

    def generate_lut_mask(self):
        """
        Generate look up table for mask (distance transform)
        :return:
        """
        # ColorTransferFunction
        colors = [(1, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]
        for idx in range(self.number_time_steps - 1):
            ctf01 = vtk.vtkColorTransferFunction()
            ctf01.AddRGBPoint(0, *self.bg_color)
            ctf01.AddRGBPoint(1, *colors[idx])
            ctf01.AddRGBPoint(255, *colors[idx + 1])
            self.ctf.append(ctf01)

        # look up table
        for idx in range(self.number_time_steps - 1):
            tableSize = 255
            lut01 = vtk.vtkLookupTable()
            lut01.SetNumberOfTableValues(tableSize)
            lut01.SetRampToLinear()
            for ii, ss in enumerate([float(xx) for xx in range(tableSize)]):
                cc = self.ctf[idx].GetColor(ss)
                # oo = otf.GetValue(ss)
                oo = 0.7
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
        lut_contour = vtk.vtkLookupTable()
        lut_contour.SetNumberOfColors(2)
        lut_contour.SetTableValue(0, 1, 1, 1, 0)
        lut_contour.SetTableValue(1, 0, 0, 0, 1)
        lut_contour.Build()
        lut_contour.GetValueRange()
        self.lut_contour = lut_contour

    def generate_actors_mask(self):
        self.actors = []
        for idx in range(self.number_time_steps - 1):
            scalarValuesToColors = vtk.vtkImageMapToColors()
            scalarValuesToColors.SetLookupTable(self.lut[idx])
            scalarValuesToColors.SetInputData(self.vtk_mask[idx])
            scalarValuesToColors.GetOutput()

            # Create an image actor
            actor = vtk.vtkImageSlice()
            actor.SetMapper(vtk.vtkImageSliceMapper())
            actor.GetMapper().SetSliceNumber(self.slice_number)
            actor.GetMapper().SetInputConnection(scalarValuesToColors.GetOutputPort())
            actor.GetProperty().SetOpacity(0.8)
            self.actors.append(actor)

    def generate_actors_contour(self):
        self.actors_contour = []
        for idx in range(self.number_time_steps):
            scalarValuesToColors = vtk.vtkImageMapToColors()
            scalarValuesToColors.SetLookupTable(self.lut_contour)
            scalarValuesToColors.SetInputData(self.vtk_contour[idx])
            scalarValuesToColors.GetOutput()

            # Create an image actor
            actor = vtk.vtkImageSlice()
            actor.SetMapper(vtk.vtkImageSliceMapper())
            actor.GetMapper().SetSliceNumber(self.slice_number)
            actor.GetMapper().SetInputConnection(scalarValuesToColors.GetOutputPort())
            actor.GetProperty().SetOpacity(0.8)
            self.actors_contour.append(actor)
