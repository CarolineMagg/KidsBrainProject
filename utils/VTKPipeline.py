########################################################################################################################
# Class to create the VTKPipeline that combines all actors, renderer, interactor and manipulations of such.
########################################################################################################################
import vtk
import os
import sys
import logging

from utils.SliceText import SliceText
from utils.TimeText import TimeText
from utils.VTKSegmentationActors import VTKSegmentationActors
from utils.VTKSegmentationMask import VTKSegmentationMask

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

__author__ = "c.magg"


class VTKPipeline:

    def __init__(self, nr_time_steps, path_dir, structure, fill=False, color_map='rgbo'):

        self.window = None

        # Read data
        self.reader = vtk.vtkPNGReader()
        path_png_dir = os.path.join(path_dir, 'png')
        path_png_files = [os.path.join(path_png_dir, x) for x in os.listdir(path_png_dir)]
        path_png_files = sorted(path_png_files, key=lambda x: int(x.split('slice')[-1].split('.')[0]))
        self.path_png_files = None
        self.UpdateReader(path_png_files)

        sn = 10
        self.dicom = vtk.vtkImageSlice()
        self.dicom.SetMapper(vtk.vtkImageSliceMapper())
        self.dicom.GetMapper().SetSliceNumber(sn)
        self.dicom.GetProperty().SetOpacity(0.5)
        self.dicom.GetMapper().SetInputConnection(self.reader.GetOutputPort())
        self.dicom.SetPosition(0, 0, 0)

        # Mask and contour actors
        self.bg_color = (1, 1, 1)
        self.fill_mask = fill
        self.color_map = color_map
        path_data_mask = [os.path.join(path_dir, x) for x in os.listdir(path_dir) if 'init' in x or 't1' in x]
        for idx, path_mask in enumerate(path_data_mask):
            path_data_mask[idx] = os.path.join(path_mask, structure)
        logging.debug('VTKPipeline: Update mask folders to {0}'.format(path_data_mask))
        vtkConverter = VTKSegmentationMask(path_data_mask, self.path_png_files, structure, fill=self.fill_mask,
                                           accuracy=False)
        vtk_mask, vtk_contour = vtkConverter.generate()
        self.bg_color = vtkConverter.bg_color
        self.accuracy = [1, 0.7, 0.5, 0.9]
        current_slice = self.dicom.GetMapper().GetSliceNumber()
        self.actorsGenerator = VTKSegmentationActors(color_map=self.color_map)
        self.actors_mask, self.actors_contour = self.actorsGenerator.UpdateActors(vtk_mask, vtk_contour, current_slice,
                                                                                  self.accuracy)

        # Slice status message
        slice_number = self.dicom.GetMapper().GetSliceNumber()
        slice_number_max = self.dicom.GetMapper().GetSliceNumberMaxValue()
        self.sliceText = SliceText(slice_number, slice_number_max)

        # Time status message
        self.timeText = TimeText(0, nr_time_steps)

        # create and add renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*self.bg_color)
        self.renderer.ResetCamera()

        self.renderer.AddViewProp(self.dicom)
        self.renderer.AddActor2D(self.sliceText.sliceTextActor)
        self.renderer.AddActor2D(self.timeText.timeTextActor)
        self.renderer.AddViewProp(self.actors_mask)
        for idx in range(len(self.actors_contour)):
            self.renderer.AddActor(self.actors_contour[idx])

        # Create interactor (customized)
        self.interactorStyle = vtk.vtkInteractorStyleImage()
        self.interactorStyle.SetInteractionModeToImageSlicing()
        # Create interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(self.interactorStyle)
        self.interactorStyle.AddObserver("MouseWheelForwardEvent", self.MoveSliceFoward)
        self.interactorStyle.AddObserver("MouseWheelBackwardEvent", self.MoveSliceBackward)

    def AddActorsToRenderer(self):
        self.renderer.AddViewProp(self.dicom)
        self.renderer.AddActor2D(self.sliceText.sliceTextActor)
        self.renderer.AddActor2D(self.timeText.timeTextActor)
        self.renderer.AddViewProp(self.actors_mask)
        for idx in range(len(self.actors_contour)):
            self.renderer.AddActor(self.actors_contour[idx])

    def RemoveDicomActor(self):
        logging.debug("VTKPipeline: Remove png actor.")
        self.renderer.RemoveViewProp(self.dicom)
        self.window.Render()

    def AddDicomActor(self):
        logging.debug("VTKPipeline: Add png actor.")
        self.renderer.RemoveViewProp(self.actors_mask)
        for idx in range(len(self.actors_contour)):
            self.renderer.RemoveViewProp(self.actors_contour[idx])
        self.renderer.AddViewProp(self.dicom)
        self.renderer.AddViewProp(self.actors_mask)
        for idx in range(len(self.actors_contour)):
            self.renderer.AddActor(self.actors_contour[idx])
        self.window.Render()

    def UpdateReader(self, new_path):
        self.path_png_files = new_path
        logging.debug("VTKPipeline: Update png reader with new first file {0}".format(new_path[0]))
        filePath = vtk.vtkStringArray()
        filePath.SetNumberOfValues(len(new_path))
        for i in range(0, len(new_path), 1):
            filePath.SetValue(i, new_path[i])
        self.reader.SetFileNames(filePath)
        self.reader.Update()

    def UpdateColorMap(self, color_map):
        self.color_map = color_map
        self.actorsGenerator.UpdateColorMap(color_map)

    def UpdateMask(self, new_path, structure, fill_toogle=None):
        for idx, path_mask in enumerate(new_path):
            new_path[idx] = os.path.join(path_mask, structure)
        if fill_toogle is not None:
            self.fill_mask = fill_toogle
        if 'Test1' in new_path:
            self.accuracy = None#[1, 0.7, 0.5, 0.9]
        elif 'Test2/Segmentation/init/Brain' in new_path[0]:
            self.accuracy = None#[1, 0.2, 0.95]  # [1, 0.2, 0.95]
        else:
            self.accuracy = None
        logging.debug('VTKPipeline: Update mask folders to {0} with fill mask {1} and colormap {2}'.format(new_path,
                                                                                                      self.fill_mask,
                                                                                                      self.color_map))
        vtkConverter = VTKSegmentationMask(new_path, self.path_png_files, structure=structure, fill=self.fill_mask)
        vtk_mask, vtk_contour = vtkConverter.generate()
        self.bg_color = vtkConverter.bg_color
        current_slice = self.dicom.GetMapper().GetSliceNumber()
        self.renderer.RemoveViewProp(self.actors_mask)
        for idx in range(len(self.actors_contour)):
            self.renderer.RemoveViewProp(self.actors_contour[idx])
        self.actorsGenerator = VTKSegmentationActors(color_map=self.color_map)
        self.actors_mask, self.actors_contour = self.actorsGenerator.UpdateActors(vtk_mask, vtk_contour, current_slice,
                                                                                  self.accuracy)
        self.renderer.AddViewProp(self.actors_mask)
        for idx in range(len(self.actors_contour)):
            self.renderer.AddActor(self.actors_contour[idx])
        self.window.Render()

    def SetWindow(self, window):
        self.window = window

    def MoveSliceFoward(self, obj, event):
        current_slice = self.dicom.GetMapper().GetSliceNumber()
        self.MoveSlice(current_slice + 1)

    def MoveSliceBackward(self, obj, event):
        current_slice = self.dicom.GetMapper().GetSliceNumber()
        self.MoveSlice(current_slice - 1)

    def MoveSlice(self, new_slice):
        new_slice = max(min(self.dicom.GetMapper().GetSliceNumberMaxValue(), new_slice),
                        self.dicom.GetMapper().GetSliceNumberMinValue())
        logging.debug('VTKPipeline: Setting new slice {0}'.format(new_slice))
        self.dicom.GetMapper().SetSliceNumber(new_slice)
        for idx in range(len(self.actors_contour)):
            self.actors_contour[idx].GetMapper().SetSliceNumber(new_slice)
            self.actors_contour[idx].Update()
        self.actors_mask.GetMapper().SetSliceNumber(new_slice)
        self.actors_mask.Update()

        self.sliceText.SetInput(new_slice, self.dicom.GetMapper().GetSliceNumberMaxValue())
        self.window.Render()

    def SetTimeText(self, new_time_step, max_time_step):
        logging.debug("VTKPipeline: Setting new time step {0}".format(new_time_step))
        self.timeText.SetInput(new_time_step, max_time_step)
        self.window.Render()
