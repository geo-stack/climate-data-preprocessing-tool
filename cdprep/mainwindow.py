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
from appconfigs.base import get_home_dir
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QToolButton, QLineEdit, QGridLayout,
    QLabel, QWidget, QFileDialog)

# ---- Local imports
from cdprep import __namever__
from cdprep.config.main import CONF
from cdprep.config.icons import get_icon, get_iconsize
from cdprep.dwnld_data.dwnld_data_manager import WeatherStationDownloader
from cdprep.gapfill_data.gapfill_weather_gui import WeatherDataGapfiller
from cdprep.utils.qthelpers import (
    create_toolbar_stretcher, qbytearray_to_hexstate, hexstate_to_qbytearray)
from cdprep.config.ospath import (
    get_select_file_dialog_dir, set_select_file_dialog_dir)


class MainWindow(QMainWindow):
    """
    This is the main window of the climate data extration tool.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(__namever__)
        self.setWindowIcon(get_icon('master'))

        self.gapfiller = WeatherDataGapfiller()
        self.data_downloader = WeatherStationDownloader(self)

        self.setCentralWidget(self.gapfiller)

        # Setup the toolbar.
        self.show_data_downloader_btn = QToolButton()
        self.show_data_downloader_btn.setIcon(get_icon('search_weather_data'))
        self.show_data_downloader_btn.setAutoRaise(True)
        self.show_data_downloader_btn.clicked.connect(
            self.data_downloader.show)

        toolbar = QToolBar()
        toolbar.setFloatable(False)
        toolbar.setMovable(False)
        toolbar.setIconSize(get_iconsize('normal'))
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        toolbar.addWidget(self.show_data_downloader_btn)
        toolbar.addWidget(create_toolbar_stretcher())
        toolbar.addWidget(self._create_workdir_manager())

        self._restore_window_geometry()
        self._restore_window_state()
        self.set_workdir(CONF.get('main', 'working_dir', get_home_dir()))

    def _create_workdir_manager(self):
        self.workdir_ledit = QLineEdit()
        self.workdir_ledit.setReadOnly(True)

        self.workdir_btn = QToolButton()
        self.workdir_btn.setIcon(get_icon('folder_open'))
        self.workdir_btn.setAutoRaise(True)
        self.workdir_btn.clicked.connect(self.select_working_directory)

        workdir_widget = QWidget()
        workdir_layout = QGridLayout(workdir_widget)
        workdir_layout.setContentsMargins(0, 0, 0, 0)
        workdir_layout.setSpacing(1)
        workdir_layout.addWidget(QLabel('Working Directory:'), 0, 0)
        workdir_layout.addWidget(self.workdir_ledit, 0, 1)
        workdir_layout.addWidget(self.workdir_btn, 0, 2)

        return workdir_widget

    def set_workdir(self, workdir):
        if osp.exists(workdir):
            self._workdir = workdir
            CONF.set('main', 'working_dir', workdir)
            self.workdir_ledit.setText(workdir)
            self.data_downloader.workdir = workdir
            self.gapfiller.set_workdir(workdir)
        else:
            self.set_workdir(get_home_dir())

    def select_working_directory(self):
        """
        Open a dialog allowing the user to select a working directory.
        """
        # Select the download folder.
        dirname = QFileDialog().getExistingDirectory(
            self, 'Choose Working Directory', self._workdir)
        if dirname:
            set_select_file_dialog_dir(dirname)
            self.set_workdir(dirname)

    # ---- Main window settings
    def _restore_window_geometry(self):
        """
        Restore the geometry of this mainwindow from the value saved
        in the config.
        """
        hexstate = CONF.get('main', 'window/geometry', None)
        if hexstate:
            hexstate = hexstate_to_qbytearray(hexstate)
            self.restoreGeometry(hexstate)
        else:
            from gwhat.config.gui import INIT_MAINWINDOW_SIZE
            self.resize(*INIT_MAINWINDOW_SIZE)

    def _save_window_geometry(self):
        """
        Save the geometry of this mainwindow to the config.
        """
        hexstate = qbytearray_to_hexstate(self.saveGeometry())
        CONF.set('main', 'window/geometry', hexstate)

    def _restore_window_state(self):
        """
        Restore the state of this mainwindow’s toolbars and dockwidgets from
        the value saved in the config.
        """
        # Then we appply saved configuration if it exists.
        hexstate = CONF.get('main', 'window/state', None)
        if hexstate:
            hexstate = hexstate_to_qbytearray(hexstate)
            self.restoreState(hexstate)

    def _save_window_state(self):
        """
        Save the state of this mainwindow’s toolbars and dockwidgets to
        the config.
        """
        hexstate = qbytearray_to_hexstate(self.saveState())
        CONF.set('main', 'window/state', hexstate)

    # ---- Qt method override/extension
    def closeEvent(self, event):
        """Qt method override to close the project before close the app."""
        self._save_window_geometry()
        self._save_window_state()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())
