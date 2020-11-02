########################################################################################################################
# Class to create the application window
########################################################################################################################
import logging
import sys
import os
from PyQt5.QtWidgets import QMainWindow, QToolBar
from PyQt5.QtWidgets import QCheckBox, QAction
from PyQt5.QtWidgets import QFrame, QVBoxLayout
from PyQt5.QtWidgets import QFileDialog, QSlider, QComboBox
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from HelpingWindow import HelpingWindow
from VTKPipeline import VTKPipeline

__author__ = "c.magg"


class MainWindow(QMainWindow):

    def __init__(self, switch=True, parent=None):
        QMainWindow.__init__(self, parent)

        init_time_step = 0
        self.switch = switch

        # Set up frame
        self.frame = QFrame()
        self.layout = QVBoxLayout()

        # VTK widget
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.layout.addWidget(self.vtkWidget)

        self.path_dir = "../../Data/Test1/Segmentation/"
        self.structure = 'Brain'
        self.time_steps = [x for x in os.listdir(self.path_dir) if 'png' in x]
        self.nr_time_steps = len(self.time_steps) - 1
        self.vtkPipeline = VTKPipeline(self.nr_time_steps, self.path_dir, self.structure, fill=False)
        # self.changeReader(nr_time_step=0)
        self.vtkPipeline.SetWindow(self.vtkWidget)
        self.renderer = self.vtkPipeline.renderer
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetUseDepthPeeling(1)
        self.renderer.SetMaximumNumberOfPeels(100)
        self.renderer.SetOcclusionRatio(0.1)
        self.vtkWidget.GetRenderWindow().SetAlphaBitPlanes(1)
        self.vtkWidget.GetRenderWindow().SetMultiSamples(0)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(self.vtkPipeline.interactorStyle)

        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        # Toolbar with Slider
        self.toolbar = QToolBar("Time slider")
        self.addToolBar(self.toolbar)
        self.createSliderToolbar()

        # Menu Bar
        bar = self.menuBar()
        new_folder = QAction("New Patient", self)
        new_folder.triggered.connect(self.openFolder)
        bar.addAction(new_folder)

        helping = QAction("Help", self)
        helping.triggered.connect(self.helping_button)
        bar.addAction(helping)

        quit = QAction("Quit", self)
        quit.triggered.connect(self.closeWindow)
        bar.addAction(quit)

        # Show and initialize
        self.show()
        self.vtkWidget.Initialize()

    def closeWindow(self, s):
        logging.debug("MainWindow: Close window.")
        self.close()

    def createSliderToolbar(self, init_value=0):
        logging.debug("MainWindow: Create slider toolbar.")
        self.removeToolBar(self.toolbar)
        self.toolbar = QToolBar("Time slider")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.timeStepChange)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.time_steps) - 1)
        if init_value > len(self.time_steps) - 1:
            init_value = len(self.time_steps) - 1
        self.slider.setValue(init_value)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        logging.debug('MainWindow: Create slider {0}/{1}'.format(init_value, len(self.time_steps) - 1))
        self.toolbar.addWidget(self.slider)

        self.toolbar.addSeparator()

        background_toggle = QCheckBox()
        background_toggle.setText("Background Toggle")
        background_toggle.setChecked(True)
        background_toggle.stateChanged.connect(self.backgroundToogleChange)
        self.toolbar.addWidget(background_toggle)

        self.toolbar.addSeparator()

        self.cb = QComboBox()
        items = os.listdir(os.path.join(self.path_dir, 'init'))
        logging.debug('MainWindow: Available structures {0}'.format(items))
        self.cb.addItems(items)
        self.cb.currentIndexChanged.connect(self.selectStructure)
        self.toolbar.addWidget(self.cb)

        self.toolbar.addSeparator()

        self.colormap = QComboBox()
        items = ['RGBO', 'Plasma', 'Viridis']
        self.colormap.addItems(items)
        self.colormap.currentIndexChanged.connect(self.selectColorMap)
        self.toolbar.addWidget(self.colormap)

        self.toolbar.addSeparator()

        fill_mask_toggle = QCheckBox()
        fill_mask_toggle.setText("Fill mask")
        fill_mask_toggle.setChecked(False)
        fill_mask_toggle.stateChanged.connect(self.fillMaskToggleChange)
        self.toolbar.addWidget(fill_mask_toggle)

        self.addToolBar(self.toolbar)

    def backgroundToogleChange(self, s):
        if s == 0:
            logging.debug('MainWindow: no background')
            self.vtkPipeline.RemoveDicomActor()
        elif s == 2:
            logging.debug('MainWindow: background')
            self.vtkPipeline.AddDicomActor()

    def selectColorMap(self, s):
        path_data_mask = [os.path.join(self.path_dir, x) for x in os.listdir(self.path_dir) if
                          'init' in x or 't1' in x]
        items = ['rgbo', 'plasma', 'viridis']
        self.vtkPipeline.UpdateColorMap(items[s])
        self.vtkPipeline.UpdateMask(path_data_mask, self.structure)
        logging.debug('MainWindow: colormap {0}'.format(items[s]))

    def fillMaskToggleChange(self, s):
        if s == 0:
            logging.debug("MainWindow: not fill mask")
            path_data_mask = [os.path.join(self.path_dir, x) for x in os.listdir(self.path_dir) if
                              'init' in x or 't1' in x]
            self.vtkPipeline.UpdateMask(path_data_mask, self.structure, fill_toogle=False)
        elif s == 2:
            logging.debug("MainWindow: fill mask")
            path_data_mask = [os.path.join(self.path_dir, x) for x in os.listdir(self.path_dir) if
                              'init' in x or 't1' in x]
            self.vtkPipeline.UpdateMask(path_data_mask, self.structure, fill_toogle=True)

    def selectStructure(self, s):
        logging.debug("MainWindow: Change structure to {0}".format(s))
        self.structure = self.cb.itemText(s)
        self.changeMask()

    def openFolder(self, s):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setViewMode(QFileDialog.Detail)

        if file_dialog.exec_():
            self.path_dir = file_dialog.selectedFiles()[0]
            logging.debug('MainWindow: Open folder {0}'.format(self.path_dir))
            self.vtkPipeline.MoveSlice(0)
            self.changeReader()
            self.vtkPipeline.UpdateColorMap('rgbo')
            self.changeMask()
            self.createSliderToolbar(init_value=self.slider.value())
            self.vtkPipeline.MoveSlice(0)

    def changeReader(self, nr_time_step=None):
        # png files
        self.time_steps = [x for x in os.listdir(self.path_dir) if 'png' in x]
        self.nr_time_steps = len(self.time_steps) - 1
        if nr_time_step is None:
            nr_time_step = min(self.nr_time_steps, self.slider.value())
            self.slider.setValue(nr_time_step)
            self.vtkPipeline.SetTimeText(self.slider.value(), self.nr_time_steps)
        path_dicom_dir = os.path.join(self.path_dir, self.time_steps[nr_time_step])
        path_dicom_files = [os.path.join(path_dicom_dir, x) for x in os.listdir(path_dicom_dir)]
        path_dicom_files = sorted(path_dicom_files, key=lambda x: int(x.split('slice')[-1].split('.')[0]))
        self.vtkPipeline.UpdateReader(path_dicom_files)
        logging.debug('MainWindow: Change png folder {0}'.format(path_dicom_dir))

    def changeMask(self):
        # segmentation files
        path_data_mask = [os.path.join(self.path_dir, x) for x in os.listdir(self.path_dir) if 'init' in x or 't1' in x]
        self.vtkPipeline.UpdateMask(path_data_mask, self.structure)
        logging.debug('MainWindow: Change mask folder {0}'.format(path_data_mask))

    def helping_button(self):
        logging.debug("MainWindow: Open help window.")
        self.helpingWindow = HelpingWindow()
        self.helpingWindow.show()

    def timeStepChange(self, s):
        logging.debug('MainWindow: Change step {0}/{1}'.format(self.slider.value(), self.nr_time_steps))
        self.vtkPipeline.SetTimeText(self.slider.value(), self.nr_time_steps)
        self.changeReader()

