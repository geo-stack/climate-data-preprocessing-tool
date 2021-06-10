# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

# ---- Standard imports
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import time
import os.path as osp

# ---- Third party imports
import gdown
import pandas as pd
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal as QSignal


# ---- Local imports
from cdprep.config.main import CONFIG_DIR
from cdprep.utils.maths import calc_dist_from_coord
from cdprep.dwnld_data.weather_stationlist import WeatherSationList

PROV_NAME_ABB = [('ALBERTA', 'AB'),
                 ('BRITISH COLUMBIA', 'BC'),
                 ('MANITOBA', 'MB'),
                 ('NEW BRUNSWICK', 'NB'),
                 ('NEWFOUNDLAND', 'NL'),
                 ('NORTHWEST TERRITORIES', 'NT'),
                 ('NOVA SCOTIA', 'NS'),
                 ('NUNAVUT', 'NU'),
                 ('ONTARIO', 'ON'),
                 ('PRINCE EDWARD ISLAND', 'PE'),
                 ('QUEBEC', 'QC'),
                 ('SASKATCHEWAN', 'SK'),
                 ('YUKON TERRITORY', 'YT')]
DATABASE_FILEPATH = osp.join(CONFIG_DIR, 'Station Inventory EN.csv')
MAX_FAILED_FETCH_TRY = 3


def fetch_stationlist_from_remote():
    url = 'https://drive.google.com/uc?id=1egfzGgzUb0RFu_EE5AYFZtsyXPfZ11y2'
    output = DATABASE_FILEPATH
    try:
        gdown.download(url, output, quiet=True)
        return True
    except Exception as e:
        print("Failed to download 'Station Inventory EN.csv' "
              "because of the following error:")
        print(e)
        return False


class WeatherStationFinder(QObject):
    sig_progress_msg = QSignal(str)
    sig_load_database_finished = QSignal(bool)

    def __init__(self, filelist=None, *args, **kwargs):
        super(WeatherStationFinder, self).__init__(*args, **kwargs)
        self._data = None

    @property
    def data(self):
        """Content of the ECCC database."""
        return self._data

    def load_database(self):
        """
        Load the climate station list from a file if it exist or else fetch it
        from ECCC Tor ftp server.
        """
        if osp.exists(DATABASE_FILEPATH):
            message = "Loading the climate station database from file."
            print(message)
            self.sig_progress_msg.emit(message)

            self._data = pd.read_csv(
                DATABASE_FILEPATH, encoding='utf-8-sig', skiprows=2,
                dtype={'DLY First Year': pd.Int64Dtype(),
                       'DLY Last Year': pd.Int64Dtype(),
                       'Latitude (Decimal Degrees)': float,
                       'Longitude (Decimal Degrees)': float,
                       'Elevation (m)': float})

            # Remove stations with no daily data.
            self._data = self._data[self._data['DLY First Year'].notna()]

            # Format provinces and set index to 'Climate ID'.
            self._data['Province'] = self._data['Province'].apply(
                lambda x: x.title())
            self._data = self._data.set_index('Climate ID', drop=True)

            self.sig_load_database_finished.emit(True)
        else:
            self.fetch_database()

    def fetch_database(self):
        """
        Fetch and read the list of climate stations with daily data
        from the ECCC Tor ftp server and save the result on disk.
        """
        message = "Fetching station list from ECCC Tor ftp server..."
        print(message)
        self.sig_progress_msg.emit(message)

        ts = time.time()
        self._data = None
        for i in range(MAX_FAILED_FETCH_TRY):
            if fetch_stationlist_from_remote() is False:
                print("Failed to fetch the database from "
                      " the ECCC server (%d/%d)."
                      % (i + 1, MAX_FAILED_FETCH_TRY))
                time.sleep(3)
            else:
                te = time.time()
                print("Station list fetched sucessfully in %0.2f sec."
                      % (te-ts))
                self.load_database()
                break
        else:
            message = "Failed to fetch the database from the ECCC server."
            print(message)
            self.sig_progress_msg.emit(message)
            self.sig_load_database_finished.emit(False)

    def get_stationlist(self, prov=None, prox=None, yrange=None):
        """
        Return a list of the stations in the ECCC database that
        fulfill the conditions specified in arguments.
        """
        stationdata = self._data.copy()
        if prov:
            stationdata = stationdata[stationdata['Province'].isin(prov)]
        if prox and not stationdata.empty:
            lat1, lon1, max_dist = prox
            lat2 = stationdata['Latitude (Decimal Degrees)'].values
            lon2 = stationdata['Longitude (Decimal Degrees)'].values
            dists = calc_dist_from_coord(lat1, lon1, lat2, lon2)
            stationdata = stationdata[dists <= max_dist]
        if yrange and not stationdata.empty:
            arr_ymin = stationdata['DLY First Year'].apply(
                lambda x: max(x, yrange[0]))
            arr_ymax = stationdata['DLY Last Year'].apply(
                lambda x: min(x, yrange[1]))
            arr_nyear = arr_ymax - arr_ymin + 1

            stationdata = stationdata[arr_nyear >= yrange[2]]

        stationdata = stationdata.reset_index()
        stationdata = stationdata[
            ['Name', 'Station ID', 'DLY First Year', 'DLY Last Year',
             'Province', 'Climate ID',
             'Latitude (Decimal Degrees)', 'Longitude (Decimal Degrees)',
             'Elevation (m)']
            ].values.tolist()
        stationlist = WeatherSationList()
        stationlist.add_stations(stationdata)
        return stationlist


if __name__ == '__main__':
    stn_browser = WeatherStationFinder()
    stn_browser.load_database()
    stnlist = stn_browser.get_stationlist(prov=['QC', 'ON'],
                                          prox=(45.40, -73.15, 25),
                                          yrange=(1960, 2015, 10))
