########################################################################################################################
# Class to create a time step actor
########################################################################################################################
import vtk

__author__ = "c.magg"


class TimeText:

    def __init__(self, current_time_step, max_time_step):
        # Time step status message
        self.timeTextProp = vtk.vtkTextProperty()
        self.timeTextProp.SetFontFamilyToCourier()
        self.timeTextProp.SetFontSize(20)
        self.timeTextProp.SetColor(128, 128, 128)
        self.timeTextProp.SetOpacity(1)
        self.timeTextProp.SetVerticalJustificationToBottom()
        self.timeTextProp.SetJustificationToLeft()

        self.timeTextMapper = vtk.vtkTextMapper()
        message = self.status_message(current_time_step, max_time_step)
        self.timeTextMapper.SetInput(message)
        self.timeTextMapper.SetTextProperty(self.timeTextProp)

        self.timeTextActor = vtk.vtkActor2D()
        self.timeTextActor.SetMapper(self.timeTextMapper)
        self.timeTextActor.SetPosition(15, 30)

    def SetInput(self, current_time_step, max_time_step):
        message = self.status_message(current_time_step, max_time_step)
        self.timeTextMapper.SetInput(message)

    @staticmethod
    def status_message(current_time_step, max_time_step):
        return "Time {0} / {1}".format(current_time_step, max_time_step)