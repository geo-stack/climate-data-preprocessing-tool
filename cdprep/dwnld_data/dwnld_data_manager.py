# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

# ---- Standard imports
from datetime import datetime
import sys
import os
import os.path as osp
from time import gmtime, sleep
from urllib.request import URLError, urlopen

# ---- Third party imports
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSignal as QSignal
from PyQt5.QtCore import pyqtSlot as QSlot
from PyQt5.QtCore import Qt, QPoint, QThread, QSize, QObject
from PyQt5.QtWidgets import (
    QWidget, QLabel, QDoubleSpinBox, QComboBox, QFrame, QGridLayout, QSpinBox,
    QPushButton, QDesktopWidget, QApplication, QFileDialog, QGroupBox, QStyle,
    QMessageBox, QProgressBar)

# ---- Local imports
from cdprep.config.icons import get_icon, get_iconsize
from cdprep.widgets.waitingspinner import QWaitingSpinner
from cdprep.dwnld_data.weather_stationlist import WeatherSationView
from cdprep.dwnld_data.weather_station_finder import (
    WeatherStationFinder, PROV_NAME_ABB)


class WaitSpinnerBar(QWidget):

    def __init__(self, parent=None):
        super(WaitSpinnerBar, self).__init__(parent)

        self._layout = QGridLayout(self)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._spinner = QWaitingSpinner(self, centerOnParent=False)

        icon = QWidget().style().standardIcon(QStyle.SP_MessageBoxCritical)
        pixmap = icon.pixmap(QSize(64, 64))
        self._failed_icon = QLabel()
        self._failed_icon.setPixmap(pixmap)
        self._failed_icon.hide()

        self._layout.addWidget(self._spinner, 1, 1)
        self._layout.addWidget(self._failed_icon, 1, 1)
        self._layout.addWidget(self._label, 2, 0, 1, 3)

        self._layout.setRowStretch(0, 100)
        self._layout.setRowStretch(3, 100)
        self._layout.setColumnStretch(0, 100)
        self._layout.setColumnStretch(2, 100)

    def set_label(self, text):
        """Set the text that is displayed next to the spinner."""
        self._label.setText(text)

    def show_warning_icon(self):
        """Stop and hide the spinner and show a critical icon instead."""
        self._spinner.hide()
        self._spinner.stop()
        self._failed_icon.show()

    def show(self):
        """Override Qt show to start waiting spinner."""
        self._spinner.show()
        self._failed_icon.hide()
        super(WaitSpinnerBar, self).show()
        self._spinner.start()

    def hide(self):
        """Override Qt hide to stop waiting spinner."""
        super(WaitSpinnerBar, self).hide()
        self._spinner.stop()


class WeatherStationBrowser(QWidget):
    """
    Widget that allows the user to browse and select ECCC climate stations.
    """

    ConsoleSignal = QSignal(str)
    staListSignal = QSignal(list)

    PROV_NAME = [x[0].title() for x in PROV_NAME_ABB]
    PROV_ABB = [x[1] for x in PROV_NAME_ABB]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stn_finder_worker = WeatherStationFinder()
        self.stn_finder_worker.sig_load_database_finished.connect(
            self.receive_load_database)
        self.stn_finder_thread = QThread()
        self.stn_finder_worker.moveToThread(self.stn_finder_thread)

        self.station_table = WeatherSationView()
        self.waitspinnerbar = WaitSpinnerBar()
        self.stn_finder_worker.sig_progress_msg.connect(
            self.waitspinnerbar.set_label)
        self.__initUI__()

        self.start_load_database()

    def __initUI__(self):
        self.setWindowTitle('Weather Stations Browser')
        self.setWindowIcon(get_icon('master'))
        self.setWindowFlags(Qt.Window)

        now = datetime.now()

        # ---- Tab Widget Search

        # ---- Proximity filter groupbox
        label_Lat = QLabel('Latitude :')
        label_Lat2 = QLabel('North')

        self.lat_spinBox = QDoubleSpinBox()
        self.lat_spinBox.setAlignment(Qt.AlignCenter)
        self.lat_spinBox.setSingleStep(0.1)
        self.lat_spinBox.setValue(0)
        self.lat_spinBox.setMinimum(0)
        self.lat_spinBox.setMaximum(180)
        self.lat_spinBox.setSuffix(u' °')
        self.lat_spinBox.valueChanged.connect(self.proximity_grpbox_toggled)

        label_Lon = QLabel('Longitude :')
        label_Lon2 = QLabel('West')

        self.lon_spinBox = QDoubleSpinBox()
        self.lon_spinBox.setAlignment(Qt.AlignCenter)
        self.lon_spinBox.setSingleStep(0.1)
        self.lon_spinBox.setValue(0)
        self.lon_spinBox.setMinimum(0)
        self.lon_spinBox.setMaximum(180)
        self.lon_spinBox.setSuffix(u' °')
        self.lon_spinBox.valueChanged.connect(self.proximity_grpbox_toggled)

        self.radius_SpinBox = QComboBox()
        self.radius_SpinBox.addItems(['25 km', '50 km', '100 km', '200 km'])
        self.radius_SpinBox.currentIndexChanged.connect(
            self.search_filters_changed)

        prox_search_grid = QGridLayout()
        row = 0
        prox_search_grid.addWidget(label_Lat, row, 1)
        prox_search_grid.addWidget(self.lat_spinBox, row, 2)
        prox_search_grid.addWidget(label_Lat2, row, 3)
        row += 1
        prox_search_grid.addWidget(label_Lon, row, 1)
        prox_search_grid.addWidget(self.lon_spinBox, row, 2)
        prox_search_grid.addWidget(label_Lon2, row, 3)
        row += 1
        prox_search_grid.addWidget(QLabel('Search Radius :'), row, 1)
        prox_search_grid.addWidget(self.radius_SpinBox, row, 2)

        prox_search_grid.setColumnStretch(0, 100)
        prox_search_grid.setColumnStretch(4, 100)
        prox_search_grid.setRowStretch(row+1, 100)
        prox_search_grid.setHorizontalSpacing(20)
        prox_search_grid.setContentsMargins(10, 10, 10, 10)  # (L, T, R, B)

        self.prox_grpbox = QGroupBox("Proximity filter")
        self.prox_grpbox.setCheckable(True)
        self.prox_grpbox.setChecked(False)
        self.prox_grpbox.toggled.connect(self.proximity_grpbox_toggled)
        self.prox_grpbox.setLayout(prox_search_grid)

        # ---- Province filter
        prov_names = ['All']
        prov_names.extend(self.PROV_NAME)
        self.prov_widg = QComboBox()
        self.prov_widg.addItems(prov_names)
        self.prov_widg.setCurrentIndex(0)
        self.prov_widg.currentIndexChanged.connect(self.search_filters_changed)

        layout = QGridLayout()
        layout.addWidget(self.prov_widg, 2, 1)
        layout.setColumnStretch(2, 100)
        layout.setVerticalSpacing(10)

        prov_grpbox = QGroupBox("Province filter")
        prov_grpbox.setLayout(layout)

        # ---- Data availability filter

        # Number of years with data

        self.nbrYear = QSpinBox()
        self.nbrYear.setAlignment(Qt.AlignCenter)
        self.nbrYear.setSingleStep(1)
        self.nbrYear.setMinimum(0)
        self.nbrYear.setValue(3)
        self.nbrYear.valueChanged.connect(self.search_filters_changed)

        subgrid1 = QGridLayout()
        subgrid1.addWidget(self.nbrYear, 0, 0)
        subgrid1.addWidget(QLabel('years of data between'), 0, 1)

        subgrid1.setHorizontalSpacing(10)
        subgrid1.setContentsMargins(0, 0, 0, 0)  # (L, T, R, B)
        subgrid1.setColumnStretch(2, 100)

        # Year range

        self.minYear = QSpinBox()
        self.minYear.setAlignment(Qt.AlignCenter)
        self.minYear.setSingleStep(1)
        self.minYear.setMinimum(1840)
        self.minYear.setMaximum(now.year)
        self.minYear.setValue(1840)
        self.minYear.valueChanged.connect(self.minYear_changed)

        label_and = QLabel('and')
        label_and.setAlignment(Qt.AlignCenter)

        self.maxYear = QSpinBox()
        self.maxYear.setAlignment(Qt.AlignCenter)
        self.maxYear.setSingleStep(1)
        self.maxYear.setMinimum(1840)
        self.maxYear.setMaximum(now.year)
        self.maxYear.setValue(now.year)
        self.maxYear.valueChanged.connect(self.maxYear_changed)

        subgrid2 = QGridLayout()
        subgrid2.addWidget(self.minYear, 0, 0)
        subgrid2.addWidget(label_and, 0, 1)
        subgrid2.addWidget(self.maxYear, 0, 2)

        subgrid2.setHorizontalSpacing(10)
        subgrid2.setContentsMargins(0, 0, 0, 0)
        subgrid2.setColumnStretch(4, 100)

        # Subgridgrid assembly

        grid = QGridLayout()

        grid.addWidget(QLabel('Search for stations with at least'), 0, 0)
        grid.addLayout(subgrid1, 1, 0)
        grid.addLayout(subgrid2, 2, 0)

        grid.setVerticalSpacing(5)
        grid.setRowStretch(0, 100)

        self.year_widg = QGroupBox("Data Availability filter")
        self.year_widg.setLayout(grid)

        # Setup the toolbar.
        self.btn_addSta = btn_addSta = QPushButton('Add')
        btn_addSta.setIcon(get_icon('add2list'))
        btn_addSta.setIconSize(get_iconsize('small'))
        btn_addSta.setToolTip('Add selected found weather stations to the '
                              'current list of weather stations.')
        btn_addSta.clicked.connect(self.btn_addSta_isClicked)

        btn_save = QPushButton('Save')
        btn_save.setIcon(get_icon('save'))
        btn_save.setIconSize(get_iconsize('small'))
        btn_save.setToolTip('Save the list of selected stations to a file.')
        btn_save.clicked.connect(self.btn_save_isClicked)

        btn_download = QPushButton('Download')

        self.btn_fetch = btn_fetch = QPushButton('Refresh')
        btn_fetch.setIcon(get_icon('refresh'))
        btn_fetch.setIconSize(get_iconsize('small'))
        btn_fetch.setToolTip("Updates the climate station database by"
                             " fetching it again from the ECCC ftp server.")
        btn_fetch.clicked.connect(self.btn_fetch_isClicked)

        toolbar_grid = QGridLayout()
        toolbar_widg = QWidget()

        for col, btn in enumerate([btn_addSta, btn_save, btn_fetch,
                                   btn_download]):
            toolbar_grid.addWidget(btn, 0, col+1)

        toolbar_grid.setColumnStretch(toolbar_grid.columnCount(), 100)
        toolbar_grid.setSpacing(5)
        toolbar_grid.setContentsMargins(0, 30, 0, 0)  # (L, T, R, B)

        toolbar_widg.setLayout(toolbar_grid)

        # ---- Left Panel

        panel_title = QLabel('<b>Weather Station Search Criteria</b>')

        left_panel = QFrame()
        left_panel_grid = QGridLayout()

        left_panel_grid.addWidget(panel_title, 0, 0)
        left_panel_grid.addWidget(self.prox_grpbox, 1, 0)
        left_panel_grid.addWidget(prov_grpbox, 2, 0)
        left_panel_grid.addWidget(self.year_widg, 3, 0)
        left_panel_grid.setRowStretch(4, 100)
        left_panel_grid.addWidget(toolbar_widg, 5, 0)

        left_panel_grid.setVerticalSpacing(20)
        left_panel_grid.setContentsMargins(0, 0, 0, 0)   # (L, T, R, B)
        left_panel.setLayout(left_panel_grid)

        # Setup the progress bar.
        self.progressbar = QProgressBar()
        self.progressbar.setValue(0)
        self.progressbar.hide()

        # Create the main grid.
        main_layout = QGridLayout(self)
        main_layout.addWidget(left_panel, 0, 0)
        main_layout.addWidget(self.station_table, 0, 1)
        main_layout.addWidget(self.waitspinnerbar, 0, 1)
        main_layout.addWidget(self.progressbar, 1, 0, 1, 2)
        main_layout.setColumnStretch(1, 100)

    @property
    def stationlist(self):
        return self.station_table.get_stationlist()

    @property
    def search_by(self):
        return ['proximity', 'province'][self.tab_widg.currentIndex()]

    @property
    def prov(self):
        if self.prov_widg.currentIndex() == 0:
            return self.PROV_ABB
        else:
            return self.PROV_ABB[self.prov_widg.currentIndex()-1]

    @property
    def lat(self):
        return self.lat_spinBox.value()

    def set_lat(self, x, silent=True):
        if silent:
            self.lat_spinBox.blockSignals(True)
        self.lat_spinBox.setValue(x)
        self.lat_spinBox.blockSignals(False)
        self.proximity_grpbox_toggled()

    @property
    def lon(self):
        return self.lon_spinBox.value()

    def set_lon(self, x, silent=True):
        if silent:
            self.lon_spinBox.blockSignals(True)
        self.lon_spinBox.setValue(x)
        self.lon_spinBox.blockSignals(False)
        self.proximity_grpbox_toggled()

    @property
    def rad(self):
        return int(self.radius_SpinBox.currentText()[:-3])

    @property
    def prox(self):
        if self.prox_grpbox.isChecked():
            return (self.lat, -self.lon, self.rad)
        else:
            return None

    @property
    def year_min(self):
        return int(self.minYear.value())

    def set_yearmin(self, x, silent=True):
        if silent:
            self.minYear.blockSignals(True)
        self.minYear.setValue(x)
        self.minYear.blockSignals(False)

    @property
    def year_max(self):
        return int(self.maxYear.value())

    def set_yearmax(self, x, silent=True):
        if silent:
            self.maxYear.blockSignals(True)
        self.maxYear.setValue(x)
        self.maxYear.blockSignals(False)

    @property
    def nbr_of_years(self):
        return int(self.nbrYear.value())

    def set_yearnbr(self, x, silent=True):
        if silent:
            self.nbrYear.blockSignals(True)
        self.nbrYear.setValue(x)
        self.nbrYear.blockSignals(False)

    # ---- Weather Station Finder Handlers
    def start_load_database(self, force_fetch=False):
        """Start the process of loading the climate station database."""
        if self.stn_finder_thread.isRunning():
            return

        self.station_table.clear()
        self.waitspinnerbar.show()

        # Start the downloading process.
        if force_fetch:
            self.stn_finder_thread.started.connect(
                self.stn_finder_worker.fetch_database)
        else:
            self.stn_finder_thread.started.connect(
                self.stn_finder_worker.load_database)
        self.stn_finder_thread.start()

    @QSlot()
    def receive_load_database(self):
        """Handles when loading the database is finished."""
        # Disconnect the thread.
        self.stn_finder_thread.started.disconnect()

        # Quit the thread.
        self.stn_finder_thread.quit()
        waittime = 0
        while self.stn_finder_thread.isRunning():
            sleep(0.1)
            waittime += 0.1
            if waittime > 15:
                print("Unable to quit the thread.")
                break
        # Force an update of the GUI.
        self.proximity_grpbox_toggled()
        if self.stn_finder_worker.data is None:
            self.waitspinnerbar.show_warning_icon()
        else:
            self.waitspinnerbar.hide()

    # ---- GUI handlers
    def show(self):
        super(WeatherStationBrowser, self).show()
        qr = self.frameGeometry()
        if self.parent():
            parent = self.parent()
            wp = parent.frameGeometry().width()
            hp = parent.frameGeometry().height()
            cp = parent.mapToGlobal(QPoint(wp/2, hp/2))
        else:
            cp = QDesktopWidget().availableGeometry().center()

        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def minYear_changed(self):
        min_yr = min_yr = max(self.minYear.value(), 1840)

        now = datetime.now()
        max_yr = now.year

        self.maxYear.setRange(min_yr, max_yr)
        self.search_filters_changed()

    def maxYear_changed(self):
        min_yr = 1840

        now = datetime.now()
        max_yr = min(self.maxYear.value(), now.year)

        self.minYear.setRange(min_yr, max_yr)
        self.search_filters_changed()

    # ---- Toolbar Buttons Handlers
    def btn_save_isClicked(self):
        ddir = os.path.join(os.getcwd(), 'weather_station_list.csv')
        filename, ftype = QFileDialog().getSaveFileName(
            self, 'Save normals', ddir, '*.csv;;*.xlsx;;*.xls')
        self.station_table.save_stationlist(filename)

    def btn_addSta_isClicked(self):
        rows = self.station_table.get_checked_rows()
        if len(rows) > 0:
            staList = self.station_table.get_content4rows(rows)
            self.staListSignal.emit(staList)
            print('Selected stations sent to list')
        else:
            print('No station currently selected')

    def btn_fetch_isClicked(self):
        """Handles when the button fetch is clicked."""
        self.start_load_database(force_fetch=True)

    # ---- Search Filters Handlers
    def proximity_grpbox_toggled(self):
        """
        Set the values for the reference geo coordinates that are used in the
        WeatherSationView to calculate the proximity values and forces a
        refresh of the content of the table.
        """
        if self.prox_grpbox.isChecked():
            self.station_table.set_geocoord((self.lat, -self.lon))
        else:
            self.station_table.set_geocoord(None)
        self.search_filters_changed()

    def search_filters_changed(self):
        """
        Search for weather stations with the current filter values and forces
        an update of the station table content.
        """
        if self.stn_finder_worker.data is not None:
            stnlist = self.stn_finder_worker.get_stationlist(
                    prov=self.prov, prox=self.prox,
                    yrange=(self.year_min, self.year_max, self.nbr_of_years))
            self.station_table.populate_table(stnlist)

    # ---- Download weather data
    def download_checked_stations_data(self):
        pass


class RawDataDownloader(QObject):
    """
    This class is used to download the raw data files from
    www.climate.weather.gc.ca and saves them automatically in
    <Project_directory>/Meteo/Raw/<station_name (Climate ID)>.

    ERRFLAG = Flag for the download of files - np.arrays
                  0 -> File downloaded successfully
                  1 -> Problem downloading the file
                  3 -> File NOT downloaded because it already exists
    """

    sig_download_finished = QSignal(list)
    sig_update_pbar = QSignal(int)
    ConsoleSignal = QSignal(str)

    def __init__(self):
        super(RawDataDownloader, self).__init__(parent=None)

        self.__stop_dwnld = False

        self.ERRFLAG = []

        # These values need to be pushed from the parent.

        self.dirname = []    # Directory where the downloaded files are saved
        self.stationID = []
        # Unique identifier for the station used for downloading the
        # data from the server
        self.climateID = []  # Unique identifier for the station
        self.yr_start = []
        self.yr_end = []
        self.StaName = []  # Common name given to the station (not unique)

    def stop_download(self):
        self.__stop_dwnld = True

    def download_data(self):
        """
        Download raw data files on a yearly basis from yr_start to yr_end.
        """

        staID = self.stationID
        yr_start = int(self.yr_start)
        yr_end = int(self.yr_end)
        StaName = self.StaName
        climateID = self.climateID

        self.ERRFLAG = np.ones(yr_end - yr_start + 1)

        print("Downloading data for station %s" % StaName)
        self.ConsoleSignal.emit(
            '''<font color=black>Downloading data from </font>
               <font color=blue>www.climate.weather.gc.ca</font>
               <font color=black> for station %s</font>''' % StaName)
        self.sig_update_pbar.emit(0)

        StaName = StaName.replace('\\', '_')
        StaName = StaName.replace('/', '_')
        dirname = osp.join(self.dirname, '%s (%s)' % (StaName, climateID))
        if not osp.exists(dirname):
            os.makedirs(dirname)

        # Data are downloaded on a yearly basis from yStart to yEnd
        downloaded_raw_datafiles = []
        for i, year in enumerate(range(yr_start, yr_end+1)):
            if self.__stop_dwnld:
                # Stop the downloading process.
                self.__stop_dwnld = False
                msg = "Downloading process for station %s stopped." % StaName
                print(msg)
                self.ConsoleSignal.emit("<font color=red>%s</font>" % msg)
                return

            # Define file and URL paths.
            fname = os.path.join(
                    dirname, "eng-daily-0101%s-1231%s.csv" % (year, year))
            url = ('http://climate.weather.gc.ca/climate_data/' +
                   'bulk_data_e.html?format=csv&stationID=' + str(staID) +
                   '&Year=' + str(year) + '&Month=1&Day=1&timeframe=2' +
                   '&submit=Download+Data')

            # Download data for that year.
            if osp.exists(fname):
                # If the file was downloaded in the same year that of the data
                # record, data will be downloaded again in case the data series
                # was not complete.

                # Get year of file last modification
                myear = osp.getmtime(fname)
                myear = gmtime(myear)[0]
                if myear == year:
                    self.ERRFLAG[i] = self.dwnldfile(url, fname)
                else:
                    self.ERRFLAG[i] = 3
                    print('    %s: Raw data file already exists for year %d.' %
                          (StaName, year))
            else:
                self.ERRFLAG[i] = self.dwnldfile(url, fname)
                print('    %s: Downloading raw data file for year %d.' %
                      (StaName, year))

            # Update UI :

            progress = (year - yr_start+1) / (yr_end+1 - yr_start) * 100
            self.sig_update_pbar.emit(int(progress))

            if self.ERRFLAG[i] == 1:                         # pragma: no cover
                self.ConsoleSignal.emit(
                    '''<font color=red>There was a problem downloading the
                         data of station %s for year %d.
                       </font>''' % (StaName, year))
            elif self.ERRFLAG[i] == 0:
                self.ConsoleSignal.emit(
                    '''<font color=black>Weather data for station %s
                         downloaded successfully for year %d.
                       </font>''' % (StaName, year))
                downloaded_raw_datafiles.append(fname)
            elif self.ERRFLAG[i] == 3:
                sleep(0.1)
                self.ConsoleSignal.emit(
                    '''<font color=green>A weather data file already existed
                         for station %s for year %d. Downloading is skipped.
                       </font>''' % (StaName, year))
                downloaded_raw_datafiles.append(fname)

        cmt = ("All raw  data files downloaded sucessfully for "
               "station %s.") % StaName
        print(cmt)
        self.ConsoleSignal.emit('<font color=black>%s</font>' % cmt)

        self.sig_update_pbar.emit(0)
        self.sig_download_finished.emit(downloaded_raw_datafiles)
        return downloaded_raw_datafiles

    def dwnldfile(self, url, fname):
        try:
            ERRFLAG = 0
            f = urlopen(url)

            # Write downloaded content to local file.
            with open(fname, 'wb') as local_file:
                local_file.write(f.read())
        except URLError as e:                                # pragma: no cover
            ERRFLAG = 1
            if hasattr(e, 'reason'):
                print('Failed to reach a server.')
                print('Reason: ', e.reason)
            elif hasattr(e, 'code'):
                print('The server couldn\'t fulfill the request.')
                print('Error code: ', e.code)

        return ERRFLAG


# %% if __name__ == '__main__'

if __name__ == '__main__':

    app = QApplication(sys.argv)

    stn_browser = WeatherStationBrowser()
    stn_browser.show()

    stn_browser.set_lat(45.40)
    stn_browser.set_lon(73.15)
    stn_browser.set_yearmin(1980)
    stn_browser.set_yearmax(2015)
    stn_browser.set_yearnbr(20)
    stn_browser.search_filters_changed()

    sys.exit(app.exec_())