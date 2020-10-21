########################################################################################################################
# Class to create slice information actor
########################################################################################################################
import vtk

__author__ = "c.magg"


class SliceText:

    def __init__(self, current_slice, max_slice):
        # Slice status message
        self.sliceTextProp = vtk.vtkTextProperty()
        self.sliceTextProp.SetFontFamilyToCourier()
        self.sliceTextProp.SetFontSize(20)
        self.sliceTextProp.SetColor(250, 128, 114)
        # self.sliceTextProp.SetOpacity(1)
        self.sliceTextProp.SetVerticalJustificationToBottom()
        self.sliceTextProp.SetJustificationToLeft()

        self.sliceTextMapper = vtk.vtkTextMapper()
        message = self.status_message(current_slice, max_slice)
        self.sliceTextMapper.SetInput(message)
        self.sliceTextMapper.SetTextProperty(self.sliceTextProp)

        self.sliceTextActor = vtk.vtkActor2D()
        self.sliceTextActor.SetMapper(self.sliceTextMapper)
        self.sliceTextActor.SetPosition(15, 10)

    def SetInput(self, current_slice, max_slice):
        message = self.status_message(current_slice, max_slice)
        self.sliceTextMapper.SetInput(message)

    @staticmethod
    def status_message(current_slice, max_slice):
        return "Slice {0} / {1}".format(current_slice, max_slice)
