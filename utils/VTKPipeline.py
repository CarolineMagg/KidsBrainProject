import vtk
import os
import sys
import logging

from utils.SliceText import SliceText
from utils.TimeText import TimeText
from utils.VTKSegmentationActors import VTKSegmentationActors
from utils.VTKSegmentationMask import VTKSegmentationMask

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class VTKPipeline:

    def __init__(self, nr_time_steps):

        self.window = None

        # Read data
        self.reader = vtk.vtkPNGReader()
        path_png_dir = "../../Data/Test/Segmentation/png/"
        path_png_files = [os.path.join(path_png_dir, x) for x in os.listdir(path_png_dir)]
        path_png_files = sorted(path_png_files, key=lambda x: int(x.split('slice')[-1].split('.')[0]))
        self.UpdateReader(path_png_files)

        sn = 10
        self.dicom = vtk.vtkImageSlice()
        self.dicom.SetMapper(vtk.vtkImageSliceMapper())
        self.dicom.GetMapper().SetSliceNumber(sn)
        self.dicom.GetProperty().SetOpacity(0.5)
        self.dicom.GetMapper().SetInputConnection(self.reader.GetOutputPort())

        # Mask and contour actors
        self.bg_color = (1, 1, 1)
        path_dir = "../../Data/Test/Segmentation/"
        path_data_mask = [os.path.join(path_dir, x) for x in os.listdir(path_dir) if 't0' in x or 't1' in x]
        logging.info('VTKPipeline: first mask folders {0}'.format(path_data_mask))
        vtkConverter = VTKSegmentationMask(path_data_mask, fill=False)
        vtk_mask, vtk_contour = vtkConverter.generate()
        self.bg_color = vtkConverter.bg_color
        self.actorsGenerator = VTKSegmentationActors(vtk_mask, vtk_contour, sn)
        self.actors_mask, self.actors_contour = self.actorsGenerator.UpdateActors()

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
        for idx in reversed(range(len(self.actors_mask))):
            self.renderer.AddViewProp(self.actors_mask[idx])
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

    def RemoveDicomActor(self):
        logging.info("VTKPipeline: Remove png actor.")
        self.renderer.RemoveViewProp(self.dicom)
        self.window.Render()

    def AddDicomActor(self):
        logging.info("VTKPipeline: Add png actor.")
        self.renderer.AddViewProp(self.dicom)
        self.window.Render()

    def UpdateReader(self, new_path):
        logging.info("VTKPipeline: Update png reader with new first file {0}".format(new_path[0]))
        filePath = vtk.vtkStringArray()
        filePath.SetNumberOfValues(len(new_path))
        for i in range(0, len(new_path), 1):
            filePath.SetValue(i, new_path[i])
        self.reader.SetFileNames(filePath)
        self.reader.Update()

    def UpdateMask(self, new_path):
        logging.info('VTKPipeline: Update mask folders to {0}'.format(new_path))
        vtkConverter = VTKSegmentationMask(new_path, fill=False)
        vtk_mask, vtk_contour = vtkConverter.generate()
        current_slice = self.dicom.GetMapper().GetSliceNumber()
        self.actors_mask, self.actors_contour = self.actorsGenerator.UpdateActors(vtk_mask, vtk_contour, current_slice)
        for idx in range(len(self.actors_mask)):
            self.actors_mask[idx].Update()
        for idx in range(len(self.actors_contour)):
            self.actors_contour[idx].Update()

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
        logging.info('VTKPipeline: Setting new slice {0}'.format(new_slice))
        self.dicom.GetMapper().SetSliceNumber(new_slice)
        slice_number = self.dicom.GetMapper().GetSliceNumber()

        for idx in range(len(self.actors_contour)):
            self.actors_contour[idx].GetMapper().SetSliceNumber(new_slice)
            self.actors_contour[idx].Update()
        for idx in range(len(self.actors_mask)):
            self.actors_mask[idx].GetMapper().SetSliceNumber(new_slice)
            self.actors_mask[idx].Update()

        self.sliceText.SetInput(new_slice, self.dicom.GetMapper().GetSliceNumberMaxValue())
        self.window.Render()

    def SetTimeText(self, new_time_step, max_time_step):
        logging.info("VTKPipeline: Setting new time step {0}".format(new_time_step))
        self.timeText.SetInput(new_time_step, max_time_step)
        self.window.Render()