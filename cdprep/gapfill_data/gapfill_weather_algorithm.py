# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Jean-Sébastien Gosselin
# Licensed under the terms of the MIT License
# (https://github.com/jnsebgosselin/pygwd)
# -----------------------------------------------------------------------------

# ---- Standard library imports
import csv
import os
import os.path as osp
from time import strftime, process_time
from copy import copy
from datetime import datetime

# ---- Third party imports
import numpy as np
import pandas as pd
from xlrd.xldate import xldate_from_date_tuple
from PyQt5.QtCore import pyqtSignal as QSignal
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication

# import statsmodels.api as sm
# import statsmodels.regression as sm_reg
# from statsmodels.regression.linear_model import OLS
# from statsmodels.regression.quantile_regression import QuantReg

# ---- Local imports
from cdprep.config.gui import RED, LIGHTGRAY
from cdprep.gapfill_data.read_weather_data import read_weather_datafile
from cdprep import __namever__

PRECIP_VARIABLES = ['Ptot']
TEMP_VARIABLES = ['Tmax', 'Tavg', 'Tmin']
VARNAMES = PRECIP_VARIABLES + TEMP_VARIABLES


class DataGapfiller(QObject):
    """
    This class manage all that is related to the gap-filling of weather data
    records, including reading the data file on the disk.

    Parameters
    ----------
    NSTAmax : int
    limitDist : float
    limitAlt : float
    regression_mode : int
    full_error_analysis : bool
    """
    sig_gapfill_progress = QSignal(int)
    sig_console_message = QSignal(str)
    sig_gapfill_finished = QSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target = None
        self.alt_and_dist = None
        self.corcoef = None

        # ---- Required Inputs
        self.time_start = None
        self.time_end = None

        self.WEATHER = self.wxdatasets = WeatherData()

        self.inputDir = None
        self.isParamsValid = False

        # ---- Define Parameters Default
        # Maximum number of neighboring stations that will be used to fill
        # the missing data in the target station
        self.NSTAmax = 4

        self.limitDist = 100
        self.limitAlt = 350

        # if *regression_mode* = 1: Ordinary Least Square
        # if *regression_mode* = 0: Least Absolute Deviations
        self.regression_mode = 1

        # Set whether a complete analysis of the estimation errors is
        # conducted with a cross-validation procedure while filling missing
        # data.
        self.full_error_analysis = False

    @property
    def outputdir(self):
        if self.inputDir is None:
            return None
        else:
            return osp.join(self.inputDir, 'GAPFILLED')

    @property
    def NSTAmax(self):
        return self.__NSTAmax

    @NSTAmax.setter
    def NSTAmax(self, x):
        if type(x) != int or x < 1:
            raise ValueError('!WARNING! NSTAmax must be must be an integer'
                             ' with a value greater than 0.')
        self.__NSTAmax = x

    def load_data(self):
        """
        Read the csv files in the input data directory folder.
        """
        if not self.inputDir:
            print('Please specify a valid input data file directory.')
            return []
        if not osp.exists(self.inputDir):
            print('Data Directory path does not exists.')
            return []

        filepaths = [
            osp.join(self.inputDir, f) for
            f in os.listdir(self.inputDir) if f.endswith('.csv')]
        print('{:d} csv files were found in {}.'.format(
            len(filepaths), self.inputDir))

        print('Loading data from csv files...')
        self.target = None
        self.alt_and_dist = None
        self.corcoef = None
        self.wxdatasets.load_and_format_data(filepaths)
        print('Data loaded successfully.')

        return self.wxdatasets.station_ids

    def get_target_station(self):
        """
        Return the metadata related to the current target station.
        """
        return self.wxdatasets.metadata.loc[self.target]

    def set_target_station(self, station_id):
        """
        Set the target station to the station corresponding to the specified
        station id.
        """
        if station_id not in self.wxdatasets.station_ids:
            self.target = None
            self.alt_and_dist = None
            self.corcoef = None
            raise ValueError("No data currently loaded for station '{}'."
                             .format(station_id))
        else:
            self.target = station_id
            self.alt_and_dist = self.wxdatasets.alt_and_dist_calc(station_id)
            self.corcoef = (
                self.wxdatasets.compute_correlation_coeff(station_id))

    def read_summary(self):
        return self.WEATHER.read_summary(self.outputdir)

    def get_valid_neighboring_stations(self):
        """
        Return the list of neighboring stations that are within the
        horizontal and altitude range of the target station.
        """
        # If cutoff limits for the horizontal distance and altitude are set
        # to a negative number, all stations are kept regardless of their
        # distance or altitude difference with the target station.
        valid_stations = self.alt_and_dist.copy()
        if self.limitDist > 0:
            valid_stations = valid_stations[
                valid_stations['hordist'] <= self.limitDist]
        if self.limitAlt > 0:
            valid_stations = valid_stations[
                valid_stations['altdiff'].abs() <= self.limitAlt]
        valid_stations = valid_stations.index.values.tolist()
        valid_stations.remove(self.target)
        return valid_stations

    def gapfill_data(self):
        tstart_total = process_time()

        neighbors = self.get_valid_neighboring_stations()
        gapfill_date_range = pd.date_range(
            start=self.time_start, end=self.time_end, freq='D')
        y2fill = pd.DataFrame(
            np.nan, index=gapfill_date_range, columns=VARNAMES)
        for varname in VARNAMES:
            # When a station does not have enough data for a given variable,
            # its correlation coefficient is set to nan. If all the stations
            # have a NeN value in the correlation table for a given variable,
            # it means there is not enough data available overall to estimate
            # and fill the missing data for that variable.
            var2fill = (self.corcoef.loc[neighbors]
                        .dropna(axis=1, how='all').columns.tolist())
            if varname not in var2fill:
                msg = ("Variable {} will not be filled because there "
                       "is not enough data.").format(varname)
                print(msg)
                self.sig_console_message.emit(
                    '<font color=red>%s</font>' % msg)
                continue
            tstart = process_time()
            print('Gapfilling data for variable {}...'.format(varname))

            reg_models = {}
            notnull = self.wxdatasets.data[varname].loc[
                gapfill_date_range, neighbors].notnull()
            notnull_groups = notnull.groupby(by=neighbors, axis=0)
            for group in notnull_groups:
                if len(group[1].columns) == 0:
                    # It is impossible to fill the data in this group
                    # because all neighboring stations are empty.
                    continue
                group_dates = group[1].index
                group_neighbors = group[1].columns[list(group[0])]

                # Determines the neighboring stations to include in the
                # regression model.
                model_neighbors = list(
                    self.corcoef.loc[group_neighbors]
                    .sort_values(varname, axis=0, ascending=False)
                    .index
                    )[:self.NSTAmax]

                neighbors_combi = ', '.join(model_neighbors)
                if neighbors_combi in reg_models:
                    # Regression coefficients and RSME are recalled
                    # from the memory matrices.
                    A = reg_models[neighbors_combi]
                else:
                    # This is the first time this neighboring stations
                    # combination is encountered in the routine,
                    # regression coefficients need to be calculated.

                    # The data for the current variable are sorted by
                    # their stations in in descending correlation
                    # coefficient.
                    YX = self.wxdatasets.data[varname][
                        [self.target] + model_neighbors].copy()

                    # Remove all rows containing at least one nan value.
                    YX = YX.dropna()

                    # Rows for which precipitation of the target station
                    # and all the neighboring stations is 0 are removed.
                    # This is only applicable for precipitation, not air
                    # temperature.
                    if varname in ['Ptot']:
                        YX = YX.loc[(YX != 0).any(axis=1)]

                    # Dependant variable (target)
                    Y = YX[self.target].values

                    # Independant variables (neighbors)
                    X = YX[model_neighbors].values

                    # Add a unitary array to X for the intercept term if
                    # variable is a temperature type data.
                    # (though this was questionned by G. Flerchinger)
                    if varname in ['Tmax', 'Tavg', 'Tmin']:
                        X = np.hstack((np.ones((len(Y), 1)), X))

                    # Generate the MLR Model
                    A = self.build_mlr_model(X, Y)

                    # Calcul the RMSE.

                    # Calculate a RMSE between the estimated and
                    # measured values of the target station.
                    # RMSE with 0 value are not accounted for
                    # in the calcultation.
                    Yp = np.dot(A, X.transpose())

                    rmse = (Y - Yp)**2          # MAE = np.abs(Y - Yp)
                    rmse = rmse[rmse != 0]      # MAE = MAE[MAE!=0]
                    rmse = np.mean(rmse)**0.5   # MAE = np.mean(MAE)
                    # print('Calcul RMSE', rmse)

                    # Store values in memory.
                    reg_models[neighbors_combi] = A

                # Calculate the missing values for the group.
                X = self.wxdatasets.data[varname].loc[
                    group_dates, model_neighbors].values
                if varname in ['Tmax', 'Tavg', 'Tmin']:
                    X = np.hstack((np.ones((len(X), 1)), X))

                Y = np.dot(A, X.transpose())
                # Limit precipitation to positive values.
                # This may happens when there is one or more negative
                # regression coefficients in A
                if varname in ['Ptot']:
                    Y[Y < 0] = 0

                # Store the results.
                y2fill.loc[group_dates, varname] = Y
            print('Data gapfilled for {} in {:0.1f} sec.'.format(
                varname, process_time() - tstart))

        # Gapfill dataset for the target station.
        gapfilled_data = pd.DataFrame([], index=gapfill_date_range)
        for varname in VARNAMES:
            # Fetch the original target data for varname.
            gapfilled_data[varname] = self.wxdatasets.data[varname].loc[
                gapfill_date_range, self.target]

            # Fill the gaps.
            isnull = gapfilled_data.index[gapfilled_data[varname].isnull()]
            gapfilled_data.loc[isnull, varname] = y2fill.loc[
                isnull, varname]

        message = (
            'Data completion for station %s completed successfully '
            'in %0.2f sec.') % (self.target, (process_time() - tstart_total))
        print(message)
        self.sig_console_message.emit('<font color=black>%s</font>' % message)

        # Save the gapfilled data to a file.

        # Add Year, Month and Day to the dataset and rename some columns.
        gapfilled_data['Year'] = gapfilled_data.index.year.astype(str)
        gapfilled_data['Month'] = gapfilled_data.index.month.astype(str)
        gapfilled_data['Day'] = gapfilled_data.index.day.astype(str)
        for varname in VARNAMES:
            gapfilled_data[varname] = gapfilled_data[varname].round(1)

        # Make sure the columns are in the right order.
        gapfilled_data = gapfilled_data[
            ['Year', 'Month', 'Day', 'Tmax', 'Tmin', 'Tavg', 'Ptot']]

        target_metadata = self.wxdatasets.metadata.loc[self.target]
        data_headers = ['Year', 'Month', 'Day', 'Max Temp (°C)',
                        'Min Temp (°C)', 'Mean Temp (°C)',
                        'Total Precip (mm)']
        fcontent = [
            ['Station Name', target_metadata['Station Name']],
            ['Province', target_metadata['Location']],
            ['Latitude (dd)', target_metadata['Latitude']],
            ['Longitude (dd)', target_metadata['Longitude']],
            ['Elevation (m)', target_metadata['Elevation']],
            ['Climate Identifier', self.target],
            [],
            ['Created by', __namever__],
            ['Created on', strftime("%d/%m/%Y")],
            [],
            data_headers
            ] + gapfilled_data.values.tolist()

        # Save the data to csv.
        if not osp.exists(self.outputdir):
            os.makedirs(self.outputdir)

        clean_target_name = (
            target_metadata['Station Name']
            .replace('\\', '_').replace('/', '_'))
        filename = '{} ({})_{}-{}.csv'.format(
            clean_target_name,
            self.target,
            str(min(gapfilled_data['Year'])),
            str(max(gapfilled_data['Year']))
            )

        filepath = osp.join(self.outputdir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerows(fcontent)

        if gapfilled_data.isnull().values.any():
            message = ("WARNING: Some missing data were not filled because "
                       "all neighboring stations were empty for that period.")
            print(message)
            self.sig_console_message.emit(
                '<font color=red>%s</font>' % message)

        self.sig_gapfill_finished.emit(True)
        return gapfilled_data

    def build_mlr_model(self, X, Y):

        if self.regression_mode == 1:  # Ordinary Least Square regression

            # http://statsmodels.sourceforge.net/devel/generated/
            # statsmodels.regression.linear_model.OLS.html

            # model = OLS(Y, X)
            # results = model.fit()
            # A = results.params

            # Using Numpy function:
            A = np.linalg.lstsq(X, Y, rcond=None)[0]

        else:  # Least Absolute Deviations regression

            # http://statsmodels.sourceforge.net/devel/generated/
            # statsmodels.regression.quantile_regression.QuantReg.html

            # http://statsmodels.sourceforge.net/devel/examples/
            # notebooks/generated/quantile_regression.html

            # model = QuantReg(Y, X)
            # results = model.fit(q=0.5)
            # A = results.params

            # Using Homemade function:
            A = L1LinearRegression(X, Y)

        return A

    @staticmethod
    def postprocess_fillinfo(staName, YX, tarStaIndx):

        # Extracts info related to the target station from <YXmFull>  and the
        # info related to the neighboring stations. Xm is for the
        # neighboring stations and Ym is for the target stations.

        Yname = staName[tarStaIndx]                       # target station name
        Xnames = np.delete(staName, tarStaIndx)     # neighboring station names

        Y = YX[:, tarStaIndx, :]                          # Target station data
        X = np.delete(YX, tarStaIndx, axis=1)        # Neighboring station data

        # Counts how many times each neigboring station was used for
        # estimating the data of the target stations.

        Xcount_var = np.sum(~np.isnan(X), axis=0)
        Xcount_tot = np.sum(Xcount_var, axis=1)

        # Removes the neighboring stations that were not used.

        indx = np.where(Xcount_tot > 0)[0]
        Xnames = Xnames[indx]
        X = X[:, indx]

        Xcount_var = Xcount_var[indx, :]
        Xcount_tot = Xcount_tot[indx]

        # Sort the neighboring stations by importance.

        indx = np.argsort(Xcount_tot * -1)
        Xnames = Xnames[indx]
        X = X[:, indx]

        return Yname, Y, Xnames, X, Xcount_var, Xcount_tot

    def generate_correlation_html_table(self, gapfill_parameters):
        """
        This function generate an HTML output to be displayed in the
        <Fill Data> tab display area after a target station has been
        selected by the user.
        """
        target_metadata = self.wxdatasets.metadata.loc[self.target]
        header_data = {
            'Latitude': target_metadata['Latitude'],
            'Longitude': target_metadata['Longitude'],
            'Altitude': target_metadata['Elevation'],
            'Data start': target_metadata['first_date'],
            'Data end': target_metadata['last_date']}
        target_info = (
            '<table border="0" cellpadding="1" cellspacing="0" align="left">')
        for field, value in header_data.items():
            target_info += '<tr>'
            target_info += '<td align="left">%s</td>' % field
            target_info += '<td align="left">&nbsp;=&nbsp;</td>'
            target_info += '<td align="left">%s</td>' % value
            target_info += '</tr>'
        target_info += '</table>'

        # Sort neighboring stations.

        # Stations best correlated with the target station are displayed toward
        # the top of the table while neighboring stations poorly correlated are
        # displayed toward the bottom.

        # Define a criteria for sorting the correlation quality
        # of the stations.

        # Generate the missing data table.
        fill_date_start = gapfill_parameters['date_start']
        fill_date_end = gapfill_parameters['date_end']
        table1 = '''
                 <p align=justify>
                   Table 1 : Number of days with missing data from
                   <b>%s</b> to <b>%s</b> for station <b>%s</b>:
                 </p>
                 ''' % (fill_date_start, fill_date_end,
                        target_metadata['Station Name'])
        table1 += '''
                  <table border="0" cellpadding="3" cellspacing="0"
                         align="center">
                    <tr>
                      <td colspan="5"><hr></td>
                    </tr>
                    <tr>
                      <td width=135 align="left">Weather Variable</td>
                      <td align="center">T<sub>max</sub></td>
                      <td align="center">T<sub>min</sub></sub></td>
                      <td align="center">T<sub>mean</sub></td>
                      <td align="center">P<sub>tot</sub></td>
                    </tr>
                    <tr>
                      <td colspan="5"><hr></td>
                    </tr>
                    <tr>
                     <td width=135 align="left">Days with<br>missing data</td>
                  '''

        datetime_start = datetime.strptime(
            gapfill_parameters['date_start'], '%d/%m/%Y')
        datetime_end = datetime.strptime(
            gapfill_parameters['date_end'], '%d/%m/%Y')
        total_nbr_data = (
            (datetime_end - datetime_start).total_seconds() / 3600 / 24 + 1)
        for var in self.wxdatasets.data.keys():
            data = self.wxdatasets.data[var][self.target]
            nbr_nan = len(data[
                (data.index >= datetime_start) &
                (data.index <= datetime_end) &
                (data.isnull())])
            nan_percent = round(nbr_nan / total_nbr_data * 100, 1)

            table1 += '''
                      <td align="center">
                      %d<br>(%0.1f %%)
                      </td>
                      ''' % (nbr_nan, nan_percent)
        table1 += '''
                  </tr>
                  <tr>
                  <td colspan="5"><hr></td>
                  </tr>
                  </table>
                  <br><br>
                  '''

        # Generate the correlation coefficient table
        table2 = table1
        table2 += '''
                  <p align="justify">
                    <font size="3">
                      Table 2 : Altitude difference, horizontal distance and
                      correlation coefficients for each meteorological
                      variables, calculated between station <b>%s</b> and its
                      neighboring stations :
                    <\font>
                  </p>
                  ''' % target_metadata['Station Name']

        # Generate the horizontal header of the table.
        table2 += '''
                  <table border="0" cellpadding="3" cellspacing="0"
                         align="center" width="100%%">
                    <tr>
                      <td colspan="9"><hr></td>
                    </tr>
                    <tr>
                      <td align="center" valign="bottom" width=30 rowspan="3">
                        #
                      </td>
                      <td align="left" valign="bottom" width=200 rowspan="3">
                        Neighboring Stations
                      </td>
                      <td width=60 align="center" valign="bottom" rowspan="3">
                        &#916;Alt.<br>(m)
                      </td>
                      <td width=60 align="center" valign="bottom" rowspan="3">
                        Dist.<br>(km)
                      </td>
                      <td align="center" valign="middle" colspan="4">
                        Correlation Coefficients
                      </td>
                    </tr>
                    <tr>
                      <td colspan="4"><hr></td>
                    </tr>
                    <tr>
                      <td width=60 align="center" valign="middle">
                        T<sub>max</sub>
                      </td>
                      <td width=60 align="center" valign="middle">
                        T<sub>min</sub>
                      </td>
                      <td width=60 align="center" valign="middle">
                        T<sub>mean</sub>
                      </td>
                      <td width=60 align="center" valign="middle">
                        P<sub>tot</sub>
                      </td>
                    </tr>
                    <tr>
                      <td colspan="9"><hr></td>
                    </tr>
                  '''
        corcoef = self.corcoef.sort_values('Ptot', axis=0, ascending=False)

        stations = corcoef.index.values.tolist()
        stations.remove(self.target)
        for i, station_id in enumerate(stations):
            color = ['transparent', LIGHTGRAY][i % 2]
            metadata = self.wxdatasets.metadata.loc[station_id]

            # Neighboring station names.
            table2 += '''
                       <tr bgcolor="%s">
                         <td align="center" valign="top">%02d</td>
                         <td valign="top">
                           %s
                         </td>
                      ''' % (color, i + 1, metadata['Station Name'])

            # Check the condition for the altitude difference.
            limit_altdiff = gapfill_parameters['limitAlt']
            altdiff = self.alt_and_dist.loc[station_id]['altdiff']
            if abs(altdiff) >= limit_altdiff and limit_altdiff >= 0:
                fontcolor = RED
            else:
                fontcolor = ''

            table2 += '''
                      <td align="center" valign="top">
                        <font color="%s">%0.1f</font>
                      </td>
                      ''' % (fontcolor, altdiff)

            # Check the condition for the horizontal distance.
            limit_hordist = gapfill_parameters['limitDist']
            hordist = self.alt_and_dist.loc[station_id]['hordist']
            if hordist >= limit_hordist and limit_hordist >= 0:
                fontcolor = RED
            else:
                fontcolor = ''

            table2 += '''
                      <td align="center" valign="top">
                        <font color="%s">%0.1f</font>
                      </td>
                      ''' % (fontcolor, hordist)
            # Add the correlation coefficients to the table.
            for var in ['Tmax', 'Tmin', 'Tavg', 'Ptot']:
                value = self.corcoef.loc[station_id, var]
                fontcolor = RED if value < 0.7 else ''
                table2 += '''
                          <td align="center" valign="top">
                            <font color="%s">%0.3f</font>
                          </td>
                          ''' % (fontcolor, value)
            table2 += '</tr>'
        table2 += '''  <tr>
                         <td colspan="8"><hr></td>
                       </tr>
                       <tr>
                         <td align="justify" colspan="8">
                         <font size="2">
                           * Correlation coefficients are set to
                           <font color="#C83737">NaN</font> for a given
                           variable if there is less than
                           <font color="#C83737">%d</font> pairs of data
                           between the target and the neighboring station.
                           </font>
                         </td>
                       </tr>
                     </table>
                     ''' % (365 // 2)
        return table2, target_info


class WeatherData(object):
    """
    This class contains all the weather data and weather station info
    that are needed for the gapfilling algorithm that is defined in the
    *GapFillWeather* class.
    """

    def __init__(self):

        self.data = None
        self.metadata = None
        self.fnames = []

    @property
    def filenames(self):
        """
        Return the list of file paths from which data were loaded.
        """
        return (self.metadata['filename'].tolist() if
                self.metadata is not None else [])

    @property
    def station_names(self):
        """
        Return the list of station names for which data are loaded in memory.
        """
        return (self.metadata['Station Name'].tolist() if
                self.metadata is not None else [])

    @property
    def station_ids(self):
        """
        Return the list of station IDs for which data are loaded in memory.
        """
        return (self.metadata.index.tolist() if
                self.metadata is not None else [])

    @property
    def datetimes(self):
        return (self.data['Ptot'].index.values if
                self.data is not None else [])

    def count(self):
        """
        Return the number of datasets that are currently loaded.
        """
        return len(self.station_ids)

    def load_and_format_data(self, paths):
        """
        Parameters
        ----------
        paths: list
            A list of absolute paths containing daily weater data files
        """
        if len(paths) == 0:
            return False

        self.fnames = [osp.basename(path) for path in paths]
        self.data = {var: pd.DataFrame([]) for var in VARNAMES}
        self.metadata = pd.DataFrame([])
        for i, path in enumerate(paths):
            try:
                sta_metadata, sta_data = read_weather_datafile(path)
            except Exception:
                print("Unable to read data from '{}'"
                      .format(osp.basename(path)))
            else:
                # Add the first and last date of the dataset to the metadata.
                sta_metadata['first_date'] = min(sta_data.index).date()
                sta_metadata['last_date'] = max(sta_data.index).date()

                # Append the metadata of this station to that of the others.
                sta_id = sta_metadata['Station ID']
                if ('Station ID' in self.metadata.columns and
                        sta_id in self.metadata['Station ID']):
                    print(("A dataset for station '{}' already exists. "
                           "Skipping reading data from '{}'."
                           ).format(sta_id, osp.basename(path)))
                    continue
                self.metadata = self.metadata.append(
                    sta_metadata, ignore_index=True)

                # Append the data of this station to that of the others.
                for name in VARNAMES:
                    self.data[name] = self.data[name].merge(
                        sta_data[[name]].rename(columns={name: sta_id}),
                        left_index=True,
                        right_index=True,
                        how='outer')

        # Make the daily time series continuous.
        for name in VARNAMES:
            self.data[name] = self.data[name].resample('1D').asfreq()

        # Set the index of the metadata.
        self.metadata = self.metadata.set_index('Station ID', drop=True)

        return True

    # ---- Utilities
    def alt_and_dist_calc(self, target_station_id):
        """
        Compute the horizontal distances in km and the altitude differences
        in m between the target station and each neighboring station.
        """
        alt_and_dist = (
            self.metadata[['Latitude', 'Longitude', 'Elevation']].copy())

        # Calcul horizontal and vertical distances of neighboring stations
        # from target.
        alt_and_dist['hordist'] = calc_dist_from_coord(
            alt_and_dist.loc[target_station_id, 'Latitude'],
            alt_and_dist.loc[target_station_id, 'Longitude'],
            alt_and_dist['Latitude'].values,
            alt_and_dist['Longitude'].values)

        alt_and_dist['altdiff'] = (
            alt_and_dist['Elevation'].values -
            alt_and_dist.loc[target_station_id, 'Elevation'])

        return alt_and_dist

    def compute_correlation_coeff(self, target_station_id):
        """
        Compute the correlation coefficients between the target
        station and the neighboring stations for each meteorological variable.
        """
        print('Compute correlation coefficients for the target station.')
        correl_target = None
        for var in VARNAMES:
            corr_matrix = self.data[var].corr(min_periods=365//2).rename(
                {target_station_id: var}, axis='columns')
            if correl_target is None:
                correl_target = corr_matrix[[var]]
            else:
                correl_target = correl_target.join(corr_matrix[[var]])
        return correl_target

    def generate_summary(self, project_folder):
        """
        Generate a summary of the weather records including all the data files
        contained in */<project_folder>/Meteo/Input*, including dates when the
        records begin and end, total number of data, and total number of data
        missing for each meteorological variable, and more.
        """

        fcontent = [['#', 'STATION NAMES', 'ClimateID',
                     'Lat. (dd)', 'Lon. (dd)', 'Alt. (m)',
                     'DATE START', 'DATE END', 'Nbr YEARS', 'TOTAL DATA',
                     'MISSING Tmax', 'MISSING Tmin', 'MISSING Tmean',
                     'Missing Precip']]

        for i in range(len(self.station_names)):
            record_date_start = '%04d/%02d/%02d' % (self.DATE_START[i, 0],
                                                    self.DATE_START[i, 1],
                                                    self.DATE_START[i, 2])

            record_date_end = '%04d/%02d/%02d' % (self.DATE_END[i, 0],
                                                  self.DATE_END[i, 1],
                                                  self.DATE_END[i, 2])

            time_start = xldate_from_date_tuple((self.DATE_START[i, 0],
                                                 self.DATE_START[i, 1],
                                                 self.DATE_START[i, 2]), 0)

            time_end = xldate_from_date_tuple((self.DATE_END[i, 0],
                                               self.DATE_END[i, 1],
                                               self.DATE_END[i, 2]), 0)

            number_data = float(time_end - time_start + 1)

            fcontent.append([i+1, self.STANAME[i],
                             self.ClimateID[i],
                             '%0.2f' % self.LAT[i],
                             '%0.2f' % self.LON[i],
                             '%0.2f' % self.ALT[i],
                             record_date_start,
                             record_date_end,
                             '%0.1f' % (number_data / 365.25),
                             number_data])

            # Missing data information for each meteorological variables.
            for var in range(len(self.VARNAME)):
                fcontent[-1].extend(['%d' % (self.NUMMISS[i, var])])

        output_path = os.path.join(
            project_folder, 'weather_datasets_summary.log')
        print(output_path)

    def read_summary(self, project_folder):
        """
        Read the content of the file generated by the method
        <generate_summary> and return the content of the file in a HTML
        formatted table
        """
        filename = os.path.join(project_folder, 'weather_datasets_summary.log')
        if not osp.exists(filename):
            return ''

        with open(filename, 'r') as f:
            reader = list(csv.reader(f, delimiter=','))[1:]

        table = '''
                <table border="0" cellpadding="3" cellspacing="0"
                 align="center">
                  <tr>
                    <td colspan="10"><hr></td>
                  </tr>
                  <tr>
                    <td align="center" valign="bottom"  width=30 rowspan="3">
                      #
                    </td>
                    <td align="left" valign="bottom" rowspan="3">
                      Station
                    </td>
                    <td align="center" valign="bottom" rowspan="3">
                      Climate<br>ID
                    </td>
                    <td align="center" valign="bottom" rowspan="3">
                      From<br>year
                    </td>
                    <td align="center" valign="bottom" rowspan="3">
                      To<br>year
                    </td>
                    <td align="center" valign="bottom" rowspan="3">
                      Nbr.<br>of<br>years
                    <td align="center" valign="middle" colspan="4">
                      % of missing data for
                    </td>
                  </tr>
                  <tr>
                    <td colspan="4"><hr></td>
                  </tr>
                  <tr>
                    <td align="center" valign="middle">
                      T<sub>max</sub>
                    </td>
                    <td align="center" valign="middle">
                      T<sub>min</sub>
                    </td>
                    <td align="center" valign="middle">
                      T<sub>mean</sub>
                    </td>
                    <td align="center" valign="middle">
                      P<sub>tot</sub>
                    </td>
                  </tr>
                  <tr>
                    <td colspan="10"><hr></td>
                  </tr>
                '''
        for i in range(len(reader)):
            color = ['transparent', '#E6E6E6']
            Ntotal = float(reader[i][9])
            TMAX = float(reader[i][10]) / Ntotal * 100
            TMIN = float(reader[i][11]) / Ntotal * 100
            TMEAN = float(reader[i][12]) / Ntotal * 100
            PTOT = float(reader[i][13]) / Ntotal * 100
            firstyear = reader[i][6][:4]
            lastyear = reader[i][7][:4]
            nyears = float(lastyear) - float(firstyear)
            table += '''
                     <tr bgcolor="%s">
                       <td align="center" valign="middle">
                         %02d
                       </td>
                       <td align="left" valign="middle">
                         <font size="3">%s</font>
                       </td>
                       <td align="center" valign="middle">
                         <font size="3">%s</font>
                       </td>
                       <td align="center" valign="middle">
                         <font size="3">%s</font>
                       </td>
                       <td align="center" valign="middle">
                         <font size="3">%s</font>
                       </td>
                       <td align="center" valign="middle">
                         <font size="3">%0.0f</font>
                       </td>
                       <td align="center" valign="middle">%0.0f</td>
                       <td align="center" valign="middle">%0.0f</td>
                       <td align="center" valign="middle">%0.0f</td>
                       <td align="center" valign="middle">%0.0f</td>
                     </tr>
                     ''' % (color[i % 2], i+1, reader[i][1], reader[i][2],
                            firstyear, lastyear, nyears,
                            TMAX, TMIN, TMEAN, PTOT)
        table += """
                   <tr>
                     <td colspan="10"><hr></td>
                   </tr>
                 </table>
                 """

        return table


def calc_dist_from_coord(lat1, lon1, lat2, lon2):
    """
    Compute the  horizontal distance in km between a location given in
    decimal degrees and a set of locations also given in decimal degrees.
    """
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    r = 6373  # r is the Earth radius in km

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return r * c


def L1LinearRegression(X, Y):
    """
    L1LinearRegression: Calculates L-1 multiple linear regression by IRLS
    (Iterative reweighted least squares)

    B = L1LinearRegression(Y,X)

    B = discovered linear coefficients
    X = independent variables
    Y = dependent variable

    Note 1: An intercept term is NOT assumed (need to append a unit column if
            needed).
    Note 2: a.k.a. LAD, LAE, LAR, LAV, least absolute, etc. regression

    SOURCE:
    This function is originally from a Matlab code written by Will Dwinnell
    www.matlabdatamining.blogspot.ca/2007/10/l-1-linear-regression.html
    Last accessed on 21/07/2014
    """

    # Determine size of predictor data.
    n, m = np.shape(X)

    # Initialize with least-squares fit.
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    BOld = np.copy(B)

    # Force divergence.
    BOld[0] += 1e-5

    # Repeat until convergence.
    while np.max(np.abs(B - BOld)) > 1e-6:

        BOld = np.copy(B)

        # Calculate new observation weights based on residuals from old
        # coefficients.
        weight = np.dot(B, X.transpose()) - Y
        weight = np.abs(weight)
        weight[weight < 1e-6] = 1e-6  # to avoid division by zero
        weight = weight**-0.5

        # Calculate new coefficients.
        Xb = np.tile(weight, (m, 1)).transpose() * X
        Yb = weight * Y

        B = np.linalg.lstsq(Xb, Yb, rcond=None)[0]

    return B


if __name__ == '__main__':
    from datetime import datetime
    gapfiller = DataGapfiller()

    # Set the input and output directory.
    gapfiller.inputDir = 'D:/gapfill_weather_data_test'

    # Load weather the data files and set the target station.
    station_names = gapfiller.load_data()
    gapfiller.set_target_station('7024627')

    # Define the plage over which data needs to be filled.
    gapfiller.time_start = datetime.strptime('1980-01-01', '%Y-%m-%d')
    gapfiller.time_end = datetime.strptime('2020-01-01', '%Y-%m-%d')

    # Set the gapfill parameters.
    gapfiller.NSTAmax = 3
    gapfiller.limitDist = 100
    gapfiller.limitAlt = 350
    gapfiller.full_error_analysis = False
    gapfiller.leave_one_out = False
    gapfiller.regression_mode = 0
    # 0 -> Least Absolute Deviation (LAD)
    # 1 -> Ordinary Least-Square (OLS)

    gapfilled_data = gapfiller.gapfill_data()
