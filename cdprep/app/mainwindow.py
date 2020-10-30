# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

from cdprep import __appname__
print('Starting {}...'.format(__appname__))

# ---- Setup the main Qt application.
import sys
from qtpy.QtWidgets import QApplication
app = QApplication(sys.argv)

# ---- Standard imports
import os.path as osp
import platform

# ---- Third parties imports
from appconfigs.base import get_home_dir
from qtpy.QtCore import Qt, QPoint
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
from cdprep.config.ospath import set_select_file_dialog_dir


class MainWindow(QMainWindow):
    """
    This is the main window of the climate data extration tool.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(__namever__)
        self.setWindowIcon(get_icon('master'))
        self.setContextMenuPolicy(Qt.NoContextMenu)

        if platform.system() == 'Windows':
            import ctypes
            myappid = 'climate_data_preprocessing_tool'  # arbitrary string
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                myappid)

        self.data_downloader = None

        # Setup the toolbar.
        self.show_data_downloader_btn = QToolButton()
        self.show_data_downloader_btn.setIcon(get_icon('search_weather_data'))
        self.show_data_downloader_btn.setAutoRaise(True)
        self.show_data_downloader_btn.setToolTip("Download Data")
        self.show_data_downloader_btn.clicked.connect(
            self.show_data_downloader)

        toolbar = QToolBar('Main')
        toolbar.setFloatable(False)
        toolbar.setMovable(False)
        toolbar.setIconSize(get_iconsize('normal'))
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        toolbar.addWidget(self.show_data_downloader_btn)
        toolbar.addWidget(create_toolbar_stretcher())
        toolbar.addWidget(self._create_workdir_manager())

        # Setup the main widget.
        self.gapfiller = WeatherDataGapfiller()
        self.setCentralWidget(self.gapfiller)

        self._restore_window_geometry()
        self._restore_window_state()
        self.set_workdir(CONF.get('main', 'working_dir', get_home_dir()))

    def _create_workdir_manager(self):
        self.workdir_ledit = QLineEdit()
        self.workdir_ledit.setReadOnly(True)

        self.workdir_btn = QToolButton()
        self.workdir_btn.setIcon(get_icon('folder_open'))
        self.workdir_btn.setAutoRaise(True)
        self.workdir_btn.setToolTip("Browse a working directory...")
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
            if self.data_downloader is not None:
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

    def show_data_downloader(self):
        """
        Show the download data dialog.
        """
        if self.data_downloader is None:
            self.data_downloader = WeatherStationDownloader(self)
            self.data_downloader.workdir = self._workdir
            self.data_downloader.show()
            qr = self.data_downloader.frameGeometry()
            wp = self.frameGeometry().width()
            hp = self.frameGeometry().height()
            cp = self.mapToGlobal(QPoint(wp/2, hp/2))
            qr.moveCenter(cp)
            self.data_downloader.move(qr.topLeft())
        self.data_downloader.show()
        self.data_downloader.activateWindow()
        self.data_downloader.raise_()

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
        self.gapfiller.close()
        if self.data_downloader is not None:
            self.data_downloader.close()
        event.accept()


def except_hook(cls, exception, traceback):
    """
    Used to override the default sys except hook so that this application
    doesn't automatically exit when an unhandled exception occurs.

    See this StackOverflow answer for more details :
    https://stackoverflow.com/a/33741755/4481445
    """
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    sys.excepthook = except_hook
    main = MainWindow()

    if platform.system() == 'Windows':
        from PyQt5.QtWidgets import QStyleFactory
        app.setStyle(QStyleFactory.create('WindowsVista'))

    main.show()
    sys.exit(app.exec_())
