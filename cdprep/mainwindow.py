# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

# ---- Standard imports
import sys
import os
import os.path as osp

# ---- Third parties imports
from qtpy.QtWidgets import QApplication, QMainWindow

# ---- Local imports
from cdprep import __namever__
from cdprep.config.icons import get_icon
from cdprep.gapfill_data.gapfill_weather_gui import WeatherDataGapfiller


class MainWindow(QMainWindow):
    """
    This is the main window of the climate data extration tool.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(__namever__)
        self.setWindowIcon(get_icon('master'))

        gapfiller = WeatherDataGapfiller()
        self.setCentralWidget(gapfiller)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())
