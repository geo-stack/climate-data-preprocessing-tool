# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

# ---- Standard library imports
import csv
import os
import os.path as osp
from time import strftime, process_time
from datetime import datetime

# ---- Third party imports
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSignal as QSignal
from PyQt5.QtCore import QObject

# import statsmodels.api as sm
# import statsmodels.regression as sm_reg
# from statsmodels.regression.linear_model import OLS
# from statsmodels.regression.quantile_regression import QuantReg

# ---- Local imports
from cdprep.utils.taskmanagers import WorkerBase, TaskManagerBase
from cdprep.config.gui import RED, LIGHTGRAY
from cdprep.gapfill_data.read_weather_data import read_weather_datafile
from cdprep import __namever__

PRECIP_VARIABLES = ['Ptot']
TEMP_VARIABLES = ['Tmax', 'Tavg', 'Tmin']
VARNAMES = PRECIP_VARIABLES + TEMP_VARIABLES


class DataGapfillManager(TaskManagerBase):
    sig_task_progress = QSignal(int)
    sig_status_message = QSignal(str)

    def __init__(self):
        super().__init__()
        worker = DataGapfillWorker()
        self.set_worker(worker)
        worker.sig_task_progress.connect(self.sig_task_progress.emit)
        worker.sig_status_message.connect(self.sig_status_message.emit)

    def count(self):
        """
        Return the number of datasets that are currently loaded in the
        gapfill data worker.
        """
        return self.worker().wxdatasets.count()

    def get_station_names(self):
        """
        Return the list of station names for which data are loaded in memory.
        """
        return self.worker().wxdatasets.station_names

    def get_station_ids(self):
        """
        Return the list of station IDs for which data are loaded in memory.
        """
        return self.worker().wxdatasets.station_ids

    def set_workdir(self, workdir):
        self.worker().inputDir = workdir

    def set_target_station(self, station_id, callback=None,
                           postpone_exec=False):
        """
        Set the target station to the station corresponding
        to the specified station id.

        Setting the target station also trigger the recalculation of the
        correlation coefficients with the neighboring stations.
        """
        self.add_task(
            'set_target_station',
            callback=callback,
            station_id=station_id)
        if not postpone_exec:
            self.run_tasks()

    def load_data(self, callback=None, postpone_exec=False):
        """Read the csv files in the input data directory folder."""
        self.add_task('load_data', callback=callback)
        if not postpone_exec:
            self.run_tasks()

    def gapfill_data(self, time_start, time_end, max_neighbors,
                     hdist_limit, vdist_limit, regression_mode,
                     callback=None, postpone_exec=False):
        """Gapfill the data of the target station."""
        self.add_task(
            'gapfill_data',
            callback=callback,
            time_start=time_start,
            time_end=time_end,
            max_neighbors=max_neighbors,
            hdist_limit=hdist_limit,
            vdist_limit=vdist_limit,
            regression_mode=regression_mode)
        if not postpone_exec:
            self.run_tasks()


class DataGapfillWorker(WorkerBase):
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
    sig_task_progress = QSignal(int)
    sig_status_message = QSignal(str)
    sig_console_message = QSignal(str)
    sig_gapfill_finished = QSignal(bool)

    def __init__(self):
        super().__init__()
        self.target = None
        self.alt_and_dist = None
        self.corcoef = None

        # ---- Required Inputs
        self.time_start = None
        self.time_end = None

        self.WEATHER = self.wxdatasets = WeatherData()
        self.wxdatasets.sig_task_progress.connect(self.sig_task_progress.emit)

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
            return
        if not osp.exists(self.inputDir):
            print('Data Directory path does not exists.')
            return

        filepaths = [
            osp.join(self.inputDir, f) for
            f in os.listdir(self.inputDir) if f.endswith('.csv')]
        print('{:d} csv files were found in {}.'.format(
            len(filepaths), self.inputDir))

        message = 'Reading data from csv files...'
        print(message)
        self.sig_status_message.emit(message)
        self.target = None
        self.alt_and_dist = None
        self.corcoef = None
        self.wxdatasets.load_and_format_data(filepaths)
        print('Data loaded successfully.')
        self.sig_status_message.emit('')

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
            station_name = self.wxdatasets.metadata.loc[
                station_id]['Station Name']

            message = ("Calculating correlation coefficients "
                       "for target station {}...".format(station_name))
            print(message)
            self.sig_status_message.emit(message)

            self.target = station_id
            self.alt_and_dist = self.wxdatasets.alt_and_dist_calc(station_id)
            self.corcoef = (
                self.wxdatasets.compute_correlation_coeff(station_id))

            print("Correlation coefficients calculated "
                  "for target station {}.".format(station_name))
            self.sig_status_message.emit('')

    def get_valid_neighboring_stations(self, hdist_limit, vdist_limit):
        """
        Return the list of neighboring stations that are within the
        horizontal and altitude range of the target station.
        """
        # If cutoff limits for the horizontal distance and altitude are set
        # to a negative number, all stations are kept regardless of their
        # distance or altitude difference with the target station.
        valid_stations = self.alt_and_dist.copy()
        if hdist_limit > 0:
            valid_stations = valid_stations[
                valid_stations['hordist'] <= hdist_limit]
        if vdist_limit > 0:
            valid_stations = valid_stations[
                valid_stations['altdiff'].abs() <= vdist_limit]
        valid_stations = valid_stations.index.values.tolist()
        valid_stations.remove(self.target)
        return valid_stations

    def gapfill_data(self, time_start, time_end, max_neighbors,
                     hdist_limit, vdist_limit, regression_mode):
        """Gapfill the data of the target station."""
        tstart_total = process_time()

        neighbors = self.get_valid_neighboring_stations(
            hdist_limit, vdist_limit)
        gapfill_date_range = pd.date_range(
            start=time_start, end=time_end, freq='D')
        y2fill = pd.DataFrame(
            np.nan, index=gapfill_date_range, columns=VARNAMES)
        self.sig_task_progress.emit(0)
        for i, varname in enumerate(VARNAMES):
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
            message = 'Gapfilling data for variable {}...'.format(varname)
            print(message)
            self.sig_status_message.emit(message)

            reg_models = {}
            notnull = self.wxdatasets.data[varname].loc[
                gapfill_date_range, neighbors].notnull()
            notnull_groups = notnull.groupby(by=neighbors, axis=0)
            for j, group in enumerate(notnull_groups):
                group_dates = group[1].index
                group_neighbors = group[1].columns[list(group[0])]
                if len(group_neighbors) == 0:
                    # It is impossible to fill the data in this group
                    # because all neighboring stations are empty.
                    continue

                # Determines the neighboring stations to include in the
                # regression model.
                model_neighbors = list(
                    self.corcoef.loc[group_neighbors]
                    .sort_values(varname, axis=0, ascending=False)
                    .index
                    )[:max_neighbors]

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
                    A = self.build_mlr_model(X, Y, regression_mode)

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
                self.sig_task_progress.emit(int(
                    (j + 1) / len(notnull_groups) * 100 / len(VARNAMES) +
                    i / len(VARNAMES) * 100))
            self.sig_task_progress.emit(int((i + 1) / len(VARNAMES) * 100))
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
        self.sig_status_message.emit(message)
        self.sig_console_message.emit('<font color=black>%s</font>' % message)

        if gapfilled_data.isnull().values.any():
            message = ("WARNING: Some missing data were not filled because "
                       "all neighboring stations were empty for that period.")
            print(message)
            self.sig_console_message.emit(
                '<font color=red>%s</font>' % message)

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

        self.sig_gapfill_finished.emit(True)
        return gapfilled_data

    def build_mlr_model(self, X, Y, regression_mode):
        """
        Build a multiple linear model using the provided independent (X) and
        dependent (y) variable data.
        """
        if regression_mode == 1:  # Ordinary Least Square regression

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

    def generate_html_summary_table(self):
        return self.wxdatasets.generate_html_summary_table()

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


class WeatherData(QObject):
    """
    This class contains all the weather data and weather station info
    that are needed for the gapfilling algorithm that is defined in the
    *GapFillWeather* class.
    """
    sig_task_progress = QSignal(int)

    def __init__(self):
        super().__init__()

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
        if self.metadata is None or self.metadata.empty:
            return []
        else:
            return self.metadata['Station Name'].tolist()

    @property
    def station_ids(self):
        """
        Return the list of station IDs for which data are loaded in memory.
        """
        if self.metadata is None or self.metadata.empty:
            return []
        else:
            return self.metadata.index.tolist()

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
        self.fnames = [osp.basename(path) for path in paths]
        self.data = {var: pd.DataFrame([]) for var in VARNAMES}
        self.metadata = pd.DataFrame([])
        if len(paths) == 0:
            return

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
            self.sig_task_progress.emit(int(
                i / len(paths) * 100))

        # Make the daily time series continuous.
        for name in VARNAMES:
            self.data[name] = self.data[name].resample('1D').asfreq()

        # Set the index of the metadata.
        self.metadata = self.metadata.set_index('Station ID', drop=True)

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
        correl_target = None
        for var in VARNAMES:
            corr_matrix = self.data[var].corr(min_periods=365//2).rename(
                {target_station_id: var}, axis='columns')
            if correl_target is None:
                correl_target = corr_matrix[[var]]
            else:
                correl_target = correl_target.join(corr_matrix[[var]])
        return correl_target

    def generate_html_summary_table(self):
        """
        Generate a Html table showing a summary of available and missing
        weather data for all the stations for which data were loaded in the
        current session.
        """
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
        for i, station_id in enumerate(self.station_ids):
            station_metadata = self.metadata.loc[station_id]
            color = ['transparent', LIGHTGRAY][i % 2]
            first_date = station_metadata['first_date']
            datetime_start = datetime(
                first_date.year, first_date.month, first_date.day)
            last_date = station_metadata['last_date']
            datetime_end = datetime(
                last_date.year, last_date.month, last_date.day)
            total_nbr_data = (
                (datetime_end - datetime_start)
                .total_seconds() / 3600 / 24 + 1)
            firstyear = datetime_start.year
            lastyear = datetime_end.year
            nyears = lastyear - firstyear + 1

            ptot_data = self.data['Ptot'][station_id]
            ptot_nan_percent = round(len(ptot_data[
                (ptot_data.index >= datetime_start) &
                (ptot_data.index <= datetime_end) &
                (ptot_data.isnull())
                ]) / total_nbr_data * 100, 1)
            tmax_data = self.data['Tmax'][station_id]
            tmax_nan_percent = round(len(tmax_data[
                (tmax_data.index >= datetime_start) &
                (tmax_data.index <= datetime_end) &
                (tmax_data.isnull())
                ]) / total_nbr_data * 100, 1)
            tmin_data = self.data['Tmax'][station_id]
            tmin_nan_percent = round(len(tmin_data[
                (tmin_data.index >= datetime_start) &
                (tmin_data.index <= datetime_end) &
                (tmin_data.isnull())
                ]) / total_nbr_data * 100, 1)
            tmean_data = self.data['Tmax'][station_id]
            tmean_nan_percent = round(len(tmean_data[
                (tmean_data.index >= datetime_start) &
                (tmean_data.index <= datetime_end) &
                (tmean_data.isnull())
                ]) / total_nbr_data * 100, 1)

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
                     ''' % (color, i+1, station_metadata['Station Name'],
                            station_id, firstyear, lastyear, nyears,
                            tmax_nan_percent, tmin_nan_percent,
                            tmean_nan_percent, ptot_nan_percent)
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
    gapfiller = DataGapfillWorker()

    # Set the input and output directory.
    gapfiller.inputDir = 'D:/choix_stations_telemetrie/weather_data'

    # Load weather the data files and set the target station.
    station_names = gapfiller.load_data()
    gapfiller.set_target_station('7050240')

    # Set the gapfill parameters.
    gapfilled_data = gapfiller.gapfill_data(
        time_start=datetime.strptime('1980-01-01', '%Y-%m-%d'),
        time_end=datetime.strptime('2020-01-01', '%Y-%m-%d'),
        hdist_limit=350,
        vdist_limit=100,
        max_neighbors=3,
        regression_mode=0)
    # 0 -> Least Absolute Deviation (LAD)
    # 1 -> Ordinary Least-Square (OLS)
