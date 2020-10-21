########################################################################################################################
# Class to create a helping window with navigation information
########################################################################################################################
from PyQt5.QtWidgets import QLabel, QMainWindow
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

__author__ = "c.magg"


class HelpingWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(HelpingWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Navigation")

        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        info = QLabel(
            "left mouse button - change window level\n" +
            "right left mouse - positional zoom\n" +
            "mouse wheele - slice through volume\n")
        info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(info)

        # quit = QAction("Quit",self)
        # quit.triggered.connect(self.closeWindow)
        # layout.addAction(quit)

    def closeWindow(self, s):
        self.close()

