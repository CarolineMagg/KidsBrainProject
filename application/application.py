########################################################################################################################
# Main file to start the application
########################################################################################################################
import os
import sys
from PyQt5.QtWidgets import QApplication

sys.path.append(os.path.abspath('../utils'))
from MainWindow import MainWindow

__author__ = "c.magg"

app = QApplication(sys.argv)
window = MainWindow(True)
window.show()
app.exec_()
