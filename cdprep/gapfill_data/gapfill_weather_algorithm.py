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
from cdprep.gapfill_data.gapfill_weather_postprocess import PostProcessErr
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
    add_ETP : bool
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

        self.outputDir = None
        self.inputDir = None

        self.STOP = False  # Flag used to stop the algorithm from a GUI
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

        # A flag to control if data are removed from the
        # dataset in the cross-validation procedure.
        self.leave_one_out = False

        self.fig_format = PostProcessErr.SUPPORTED_FIG_FORMATS[0]
        self.fig_language = PostProcessErr.SUPPORTED_LANGUAGES[0]

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
        return self.WEATHER.read_summary(self.outputDir)

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

    def fill_data2(self):
        # If some missing data can't be completed because all the neighboring
        # stations are empty, flag_nan is set to True and a comment is
        # issued at the end of the completion process.
        flag_nan = False

        neighbors = self.get_valid_neighboring_stations()
        gapfill_date_range = pd.date_range(
            start=self.time_start, end=self.time_end, freq='D')
        y2fill = pd.DataFrame(
            np.nan, index=gapfill_date_range, columns=VARNAMES)
        for j, varname in enumerate(VARNAMES):
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
            
            isnull = self.wxdatasets.data[varname][neighbors].isnull()

            reg_models = {}
            for i, date in enumerate(gapfill_date_range):
                # Determine neighboring stations with data for that date.
                X_row = self.wxdatasets.data[varname].loc[date]
                neighbors_with_data = sorted(list(
                    X_row[neighbors].dropna().index))
                if len(neighbors_with_data) == 0:
                    # It is impossible to fill variable because all
                    # neighboring stations are empty.
                    flag_nan = True
                    continue

                # Determines the neighboring stations to include in the
                # regression model.
                neighbors_with_data = list(
                    self.corcoef.loc[neighbors_with_data]
                    .sort_values(varname, axis=0, ascending=False)
                    .index
                    )[:self.NSTAmax]
                neighbors_combi = ', '.join(neighbors_with_data)
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
                        [self.target] + neighbors_with_data].copy()

                    # Removes row for which a data is missing in the
                    # target station data series
                    YX = YX.dropna(subset=[self.target])
                    ntot = len(YX)

                    # All rows containing at least one nan for the
                    # neighboring stations are removed
                    YX = YX.dropna()
                    nreg = len(YX)

                    # Rows for which precipitation of the target station
                    # and all the neighboring stations is 0 are removed.
                    # This is only applicable for precipitation, not air
                    # temperature.
                    if varname in ['Ptot']:
                        YX = YX.loc[(YX != 0).any(axis=1)]

                    # Dependant variable (target)
                    Y = YX[self.target].values

                    # Independant variables (neighbors)
                    X = YX[neighbors_with_data].values

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

                    RMSE = (Y - Yp)**2          # MAE = np.abs(Y - Yp)
                    RMSE = RMSE[RMSE != 0]      # MAE = MAE[MAE!=0]
                    RMSE = np.mean(RMSE)**0.5   # MAE = np.mean(MAE)

                    # Store values in memory.
                    reg_models[neighbors_combi] = A

                # Calculate the missing value of Y at row.
                X_row = X_row[neighbors_with_data].values
                if varname in ['Tmax', 'Tavg', 'Tmin']:
                    X_row = np.hstack((1, X_row))
                Y_row = np.dot(A, X_row)

                # Limit precipitation to positive values.
                # This may happens when there is one or more negative
                # regression coefficients in A
                if varname in ['Ptot']:
                    Y_row = max(Y_row, 0)

                # Store the results.
                y2fill.iat[i, j] = Y_row
            print('Data gapfilled for {} in {:0.1f} sec.'.format(
                varname, process_time() - tstart))
        return y2fill

    def fill_data(self):
        """
        Fill the missing data for the target station.
        """
        tstart = process_time()

        msg = "Data completion for station '{}' started.".format(self.target)
        print('{}\n{}\n{}'.format('-' * 50, msg, '-' * 50))
        self.sig_console_message.emit('<font color=black>%s</font>' % msg)

        # ---- Init Container Matrices

        # Save the weather data series of the target station in a new
        # 2D matrix named 'y2fill'. The NaN values contained in this matrix
        # will be filled during the data completion process

        y2fill = (self.WEATHER.data[VARNAMES[0]][[self.target]]
        # When the option 'full_error_analysis' is True, an additional empty
        # 2D matrix named 'YpFULL' is created. This matrix will be completely
        # filled with estimated data during the gapfilling process. The
        # content of this matrix will be used to produce the '.err' file.
                  .rename(columns={self.target: VARNAMES[0]}))
        for var in VARNAMES[1:]:
            y2fill = y2fill.merge(
                self.WEATHER.data[var][[self.target]]
                .rename(columns={self.target: var}),
                left_index=True, right_index=True, how='outer'
                )

        yxmfill = {}
        for var in VARNAMES:
            yxmfill[var] = pd.DataFrame(
                np.nan,
                index=self.WEATHER.data[var].index,
                columns=self.WEATHER.data[var].columns)

        log_rmse = pd.DataFrame(
            np.nan, index=y2fill.index, columns=y2fill.columns)
        log_ndat = log_rmse.copy()

        if self.full_error_analysis:
            print('\n!A full error analysis will be performed!\n')
            ypfull = log_rmse.copy()
            yxmfull = {var: yxmfill[var].copy() for var in VARNAMES}

        # Remove the neighboring stations that do not respect the distance
        # or altitude difference cutoff criteria. If cutoff limits are set
        # to a negative number, all stations are kept regardless of their
        # distance or altitude difference with the target station.
        valid_stations = self.alt_and_dist.copy()
        if self.limitDist > 0:
            valid_stations = valid_stations[
                valid_stations['hordist'] <= self.limitDist]
        if self.limitAlt > 0:
            valid_stations = valid_stations[
                valid_stations['altdiff'].abs() <= self.limitAlt]

        # Checks variables with enough data.

        # When a station does not have enough data for a given variable,
        # its correlation coefficient is set to nan. If all the stations
        # have a value of nan in the correlation table for a given variable,
        # it means there is not enough data available overall to estimate
        # and fill missing data for it.
        var2fill = []
        for var in VARNAMES:
            var_corcoef = self.corcoef.loc[valid_stations.index, var].values
            if np.sum(~np.isnan(var_corcoef)) > 1:
                var2fill.append(var)
            else:
                msg = ("!Variable %s will not be filled because there " +
                       "is not enough data!") % var
                print(msg)
                self.sig_console_message.emit(
                    '<font color=red>%s</font>' % msg)

        # Init gapfill loop

        # If some missing data can't be completed because all the neighboring
        # stations are empty, a flag is raised and a comment is issued at the
        # end of the completion process.
        flag_nan = False

        nbr_nan_total = np.sum(np.isnan(
            y2fill.loc[(y2fill.index >= self.time_start) & 
                       (y2fill.index <= self.time_end)]).values)

        # Initi the variable for the progression of the routine.

        # *progress_total* and *fill_progress* are used to display the
        # progression of the gap-filling procedure on a UI progression bar.

        if self.full_error_analysis:
            progress_total = np.size(y2fill[var2fill])
        else:
            progress_total = np.copy(nbr_nan_total)
        fill_progress = 0

        # Init variable for the .log file
        avg_rmse = {var: 0 for var in VARNAMES}
        avg_nsta = {var: 0 for var in VARNAMES}

        for var in var2fill:
            # iterates over all the weather variables with enough
            # measured data.

            print('Data completion for variable %s in progress...' % var)

            # ---- Memory Variables ---- #

            colm_memory = []      # Column sequence memory matrix
            RegCoeff_memory = []  # Regression coefficient memory matrix
            RMSE_memory = []      # RMSE memory matrix
            Ndat_memory = []      # Nbr. of data used for the regression

            # Sort station in descending correlation coefficient order.
            # The index of the target station is forced at the beginning of
            # the list.
            sorted_stations = self.sort_sta_corrcoef(var)

            # Data for the current weather variable are stored in a
            # 2D matrix where the rows are the daily weather data and the
            # columns are the weather stations.
            YX = self.WEATHER.data[var][sorted_stations].copy()

            # # Finds rows where data are missing between the date limits
            # # at the time indexes <index_start> and <index_end>.
            row_nan = YX.loc[
                (YX.index >= self.time_start) & (YX.index <= self.time_end),
                self.target
                ]
            row_nan = row_nan[row_nan.isna()].index

            # counter used in the calculation of average RMSE and NSTA values.
            it_avg = 0

            if self.full_error_analysis:
                # All the data of the time series between the specified
                # time indexes will be estimated.
                row2fill = YX.loc[
                    (YX.index >= self.time_start) & (YX.index <= self.time_end)
                    ].index
            else:
                row2fill = row_nan.copy()

            for row in row2fill:
                # Iterates over all the days with missing values.

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # This block of code is used only to stop the gap-filling
                # routine from a UI.
                if self.STOP is True:
                    msg = ('Completion process for station %s stopped.' %
                           self.target)
                    print(msg)
                    self.sig_console_message.emit(
                        '<font color=red>%s</font>' % msg)
                    self.STOP = False
                    self.sig_gapfill_finished.emit(False)
                    return
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # Find neighboring stations with valid entries at
                # row <row> in <YX>.

                row_data = YX.loc[row]
                row_data = row_data[row_data.notnull()]
                if len(row_data) == 0:
                    # It is impossible to fill variable because all
                    # neighboring stations are empty.
                    if self.full_error_analysis:
                        ypfull.loc[row, var] = np.nan
                    if row in row_nan:
                        y2fill.loc[row, var] = np.nan
                        # We set flag_nan to True so that a warning comment
                        # is issued at the end of the the completion process.
                        flag_nan = True
                else:
                    # Determines the neighboring stations to
                    # include in the regression model.
                    nsta = min(len(row_data), self.NSTAmax)
                    neighbors = row_data.index[:nsta].tolist()

                    # Adds the target station at index 0.
                    neighbors.insert(0, self.target)

                    # Store the values of the independent variables
                    # (neighboring stations) for this row in a new array.
                    # An intercept term is added if var is a temperature based
                    # variable, but not if it is precipitation type.
                    X_row = YX.loc[row, neighbors[1:]].values
                    if var in TEMP_VARIABLES:
                        X_row = np.hstack((1, X_row))

                    # A check is made to see if the current combination
                    # of neighboring stations has been encountered
                    # previously in the routine. Regression coefficients
                    # are calculated only once for a given neighboring
                    # station combination.

                    if neighbors not in colm_memory:
                        # Thi is the first time this neighboring station
                        # combination is encountered in the routine,
                        # regression coefficients need to be calculated.
                        #
                        # Note that the memory is activated only if the option
                        # 'full_error_analysis' is not active. Otherwise, the
                        # memory remains empty and a new MLR model is built
                        # for each value of the data series.
                        if self.leave_one_out is False:
                            colm_memory.append(neighbors)

                        # The data for the current variable are sorted by
                        # their stations in in descending correlation
                        # coefficient.
                        YXcolm = YX.copy()[neighbors]

                        # Force the value of the target station to a nan value
                        # for this row. This should only have an impact when
                        # the option "full_error_analysis" is activated. This
                        # is to actually remove the data being estimated from
                        # the dataset like it should properly be done in a
                        # cross-validation procedure.
                        if self.leave_one_out is False:
                            YXcolm.loc[row, self.target] = np.nan

                        # Removes row for which a data is missing in the
                        # target station data series
                        YXcolm = YXcolm.dropna(subset=[self.target])
                        ntot = len(YXcolm)

                        # All rows containing at least one nan for the
                        # neighboring stations are removed
                        YXcolm = YXcolm.dropna()
                        nreg = len(YXcolm)

                        Ndat = '%d/%d' % (nreg, ntot)

                        # Rows for which precipitation of the target station
                        # and all the neighboring stations is 0 are removed.
                        # Only applicable for precipitation, not air
                        # temperature.
                        if var in PRECIP_VARIABLES:
                            YXcolm = YXcolm.loc[(YXcolm != 0).any(axis=1)]

                        # Dependant variable (target)
                        Y = YXcolm[self.target].values
                        # Independant variables (neighbors)
                        X = YXcolm[neighbors[1:]].values

                        # Add a unitary array to X for the intercept term if
                        # variable is a temperature type data.
                        # (though this was questionned by G. Flerchinger)

                        if var in TEMP_VARIABLES:
                            X = np.hstack((np.ones((len(Y), 1)), X))

                        # Generate the MLR Model
                        A = self.build_mlr_model(X, Y)

                        # Calcul the RMSE.

                        # Calculate a RMSE between the estimated and
                        # measured values of the target station.
                        # RMSE with 0 value are not accounted for
                        # in the calcultation.
                        Yp = np.dot(A, X.transpose())

                        RMSE = (Y - Yp)**2          # MAE = np.abs(Y - Yp)
                        RMSE = RMSE[RMSE != 0]      # MAE = MAE[MAE!=0]
                        RMSE = np.mean(RMSE)**0.5   # MAE = np.mean(MAE)

                        # Add values to memory.
                        RegCoeff_memory.append(A)
                        RMSE_memory.append(RMSE)
                        Ndat_memory.append(Ndat)

                    else:
                        # Regression coefficients and RSME are recalled
                        # from the memory matrices.
                        index_memory = colm_memory.index(neighbors)
                        A = RegCoeff_memory[index_memory]
                        RMSE = RMSE_memory[index_memory]
                        Ndat = Ndat_memory[index_memory]

                    # Calculate the missing value of Y at row.
                    Y_row = np.dot(A, X_row)

                    # Limit precipitation based variable to positive values.
                    # This may happens when there is one or more negative
                    # regression coefficients in A
                    if var in PRECIP_VARIABLES:
                        Y_row = max(Y_row, 0)

                    # Store the results.
                    log_rmse.loc[row, var] = RMSE
                    log_ndat.loc[row, var] = Ndat
                    if self.full_error_analysis:
                        ypfull.loc[row, var] = Y_row

                        # Get the measured value for the target station.
                        ym_row = self.WEATHER.data[var].loc[row, self.target]

                        # There is a need to take into account that a intercept
                        # term has been added for temperature-like variables.
                        yxmfull[var].loc[row, self.target] = ym_row
                        if var in TEMP_VARIABLES:
                            yxmfull[var].loc[row, neighbors[1:]] = X_row[1:]
                        else:
                            yxmfull[var].loc[row, neighbors[1:]] = X_row
                    if row in row_nan:
                        y2fill.loc[row, var] = Y_row

                        yxmfill[var].loc[row, self.target] = Y_row
                        if var in TEMP_VARIABLES:
                            yxmfill[var].loc[row, neighbors[1:]] = X_row[1:]
                        else:
                            yxmfill[var].loc[row, neighbors[1:]] = X_row

                        avg_rmse[var] += RMSE
                        avg_nsta[var] += nsta
                        it_avg += 1
                        break
                fill_progress += 1
                self.sig_gapfill_progress.emit(
                    fill_progress / progress_total * 100)

            # Calculate estimation error for this variable.
            if it_avg > 0:
                avg_rmse[var] /= it_avg
                avg_nsta[var] /= it_avg
            else:
                avg_rmse[var] = np.nan
                avg_nsta[var] = np.nan
            print('Missing data filled for variable %s.' % var)

        # ---- End of routine

        msg = ('Data completion for station %s completed successfully ' +
               'in %0.2f sec.'
               ) % (self.target, (process_time() - tstart))
        self.sig_console_message.emit('<font color=black>%s</font>' % msg)
        print('\n' + msg)
        print('Saving data to files...')
        print('--------------------------------------------------')

        if flag_nan:
            msg = ("WARNING: Some missing data were not filled because all "
                   "neighboring stations were empty for that period.")
            print(msg)
            self.sig_console_message.emit('<font color=red>%s</font>' % msg)

        # =====================================================================
        # WRITE DATA TO FILE
        # =====================================================================

        target_data = self.WEATHER.metadata.loc[self.target]

        # Setup the directory where results are to be saved.
        clean_target_name = (target_data['Station Name']
                             .replace('\\', '_')
                             .replace('/', '_')
                             )
        folder_name = "{} ({})".format(clean_target_name, self.target)
        dirname = osp.join(self.outputDir, folder_name)
        if not osp.exists(dirname):
            os.makedirs(dirname)

        # Setup the file header.
        fheader = [
            ['Station Name', target_data['Station Name']],
            ['Station ID', target_data.name],
            ['Location', target_data['Location']],
            ['Latitude', target_data['Latitude']],
            ['Longitude', target_data['Longitude']],
            ['Elevation', target_data['Elevation']],
            [],
            ['Created by', __namever__],
            ['Created on', strftime("%d/%m/%Y")],
            []]

        return

        # Prepare .log file content.

        # Info Data Post-Processing :

        # XYinfo = self.postprocess_fillinfo(STANAME, YXmFILL, tarStaIndx)
        # Yname, Ypre = XYinfo[0], XYinfo[1]
        # Xnames, Xmes = XYinfo[2], XYinfo[3]
        # Xcount_var, Xcount_tot = XYinfo[4], XYinfo[5]

        # Yname: name of the target station
        # Ypre: Value predicted with the model for the target station
        # Xnames: names of the neighboring station to estimate Ypre
        # Xmes: Value of the measured data used to predict Ypre
        # Xcount_var: Number of times each neighboring station was used to
        #             predict Ypre, weather variable wise.
        # Xcount_tot: Number of times each neighboring station was used to
        #             predict Ypre for all variables.

        # ---- Gap-Fill Info Summary ----

        record_date_start = '%04d/%02d/%02d' % (YEAR[index_start],
                                                MONTH[index_start],
                                                DAY[index_start])

        record_date_end = '%04d/%02d/%02d' % (YEAR[index_end],
                                              MONTH[index_end],
                                              DAY[index_end])

        fcontent = copy(HEADER)
        fcontent.extend([['*** FILL PROCEDURE INFO ***'], []])
        if self.regression_mode is True:
            fcontent.append(['MLR model', 'Ordinary Least Square'])
        elif self.regression_mode is False:
            fcontent.append(['MLR model', 'Least Absolute Deviations'])
        fcontent.extend([['Precip correction', 'Not Available'],
                         ['Wet days correction', 'Not Available'],
                         ['Max number of stations', str(self.NSTAmax)],
                         ['Cutoff distance (km)', str(limitDist)],
                         ['Cutoff altitude difference (m)', str(limitAlt)],
                         ['Date Start', record_date_start],
                         ['Date End', record_date_end],
                         [], [],
                         ['*** SUMMARY TABLE ***'],
                         [],
                         ['CLIMATE VARIABLE', 'TOTAL MISSING',
                          'TOTAL FILLED', '', 'AVG. NBR STA.', 'AVG. RMSE',
                          '']])
        fcontent[-1].extend(Xnames)

        # ---- Missing Data Summary ----

        total_nbr_data = index_end - index_start + 1
        nbr_fill_total = 0
        nbr_nan_total = 0
        for var in range(nVAR):

            nbr_nan = np.isnan(DATA[index_start:index_end+1, tarStaIndx, var])
            nbr_nan = float(np.sum(nbr_nan))

            nbr_nan_total += nbr_nan

            nbr_nofill = np.isnan(Y2fill[index_start:index_end+1, var])
            nbr_nofill = np.sum(nbr_nofill)

            nbr_fill = nbr_nan - nbr_nofill

            nbr_fill_total += nbr_fill

            nan_percent = round(nbr_nan / total_nbr_data * 100, 1)
            if nbr_nan != 0:
                nofill_percent = round(nbr_nofill / nbr_nan * 100, 1)
                fill_percent = round(nbr_fill / nbr_nan * 100, 1)
            else:
                nofill_percent = 0
                fill_percent = 100

            nbr_nan = '%d (%0.1f %% of total)' % (nbr_nan, nan_percent)

            nbr_nofill = '%d (%0.1f %% of missing)' % (nbr_nofill,
                                                       nofill_percent)

            nbr_fill_txt = '%d (%0.1f %% of missing)' % (nbr_fill,
                                                         fill_percent)

            fcontent.append([VARNAME[var], nbr_nan, nbr_fill_txt, '',
                             '%0.1f' % AVG_NSTA[var],
                             '%0.2f' % AVG_RMSE[var], ''])

            for i in range(len(Xnames)):
                if nbr_fill == 0:
                    pc = 0
                else:
                    pc = Xcount_var[i, var] / float(nbr_fill) * 100
                fcontent[-1].append('%d (%0.1f %% of filled)' %
                                    (Xcount_var[i, var], pc))

        # ---- Total Missing ----

        pc = nbr_nan_total / (total_nbr_data * nVAR) * 100
        nbr_nan_total = '%d (%0.1f %% of total)' % (nbr_nan_total, pc)

        # ---- Total Filled ----

        try:
            pc = nbr_fill_total/nbr_nan_total * 100
        except TypeError:
            pc = 0
        nbr_fill_total_txt = '%d (%0.1f %% of missing)' % (nbr_fill_total, pc)

        fcontent.extend([[],
                         ['TOTAL', nbr_nan_total, nbr_fill_total_txt,
                          '', '---', '---', '']])

        for i in range(len(Xnames)):
            pc = Xcount_tot[i] / nbr_fill_total * 100
            text2add = '%d (%0.1f %% of filled)' % (Xcount_tot[i], pc)
            fcontent[-1].append(text2add)

        # ---- Info Detailed ----

        fcontent.extend([[], [],
                         ['*** DETAILED REPORT ***'],
                         [],
                         ['VARIABLE', 'YEAR', 'MONTH', 'DAY', 'NBR STA.',
                          'Ndata', 'RMSE', Yname]])
        fcontent[-1].extend(Xnames)

        for var in var2fill:
            for row in range(index_start, index_end+1):

                yp = Ypre[row, var]
                ym = DATA[row, tarStaIndx, var]
                xm = ['' if np.isnan(i) else '%0.1f' % i for i in
                      Xmes[row, :, var]]
                nsta = len(np.where(~np.isnan(Xmes[row, :, var]))[0])

                # Write the info only if there is a missing value in
                # the data series of the target station.

                if np.isnan(ym):
                    fcontent.append([VARNAME[var],
                                     '%d' % YEAR[row],
                                     '%d' % MONTH[row],
                                     '%d' % DAY[row],
                                     '%d' % nsta,
                                     '%s' % log_Ndat[row, var],
                                     '%0.2f' % log_RMSE[row, var],
                                     '%0.1f' % yp])
                    fcontent[-1].extend(xm)

        # ---- Save File ----

        YearStart = str(int(YEAR[index_start]))
        YearEnd = str(int(YEAR[index_end]))

        fname = '%s (%s)_%s-%s.log' % (clean_tarStaName,
                                       target_station_clim,
                                       YearStart, YearEnd)

        output_path = os.path.join(dirname, fname)
        self.save_content_to_file(output_path, fcontent)
        self.sig_console_message.emit(
            '<font color=black>Info file saved in %s.</font>' % output_path)

        # ------------------------------------------------------ .out file ----

        # Prepare Header :

        fcontent = copy(HEADER)
        fcontent.append(['Year', 'Month', 'Day'])
        fcontent[-1].extend(VARNAME)

        # Add Data :

        for row in range(index_start, index_end+1):
            fcontent.append(['%d' % YEAR[row],
                             '%d' % MONTH[row],
                             '%d' % DAY[row]])

            y = ['%0.1f' % i for i in Y2fill[row, :]]
            fcontent[-1].extend(y)

        # Save Data :

        fname = '%s (%s)_%s-%s.out' % (clean_tarStaName,
                                       target_station_clim,
                                       YearStart, YearEnd)

        output_path = os.path.join(dirname, fname)
        self.save_content_to_file(output_path, fcontent)

        msg = 'Meteo data saved in %s.' % output_path
        self.sig_console_message.emit('<font color=black>%s</font>' % msg)

        if self.full_error_analysis:

            # ---- Info Data Post-Processing ----

            XYinfo = self.postprocess_fillinfo(STANAME, YXmFULL, tarStaIndx)
            Yname, Ym = XYinfo[0], XYinfo[1]
            Xnames, Xmes = XYinfo[2], XYinfo[3]

            # ---- Prepare Header ----

            fcontent = copy(HEADER)
            fcontent.append(['', '', '', '', '', '',
                             'Est. Err.', Yname, Yname])
            fcontent[-1].extend(Xnames)
            fcontent.append(['VARIABLE', 'YEAR', 'MONTH', 'DAY', 'Ndata',
                             'RMSE', 'Ypre-Ymes', 'Ypre', 'Ymes'])
            for i in range(len(Xnames)):
                fcontent[-1].append('X%d' % i)

            # ---- Add Data to fcontent ----

            for var in range(nVAR):
                for row in range(index_start, index_end+1):

                    yp = YpFULL[row, var]
                    ym = Ym[row, var]
                    xm = ['' if np.isnan(i) else '%0.1f' % i for i in
                          Xmes[row, :, var]]

                    # Write the info only if there is a measured value in
                    # the data series of the target station.

                    if not np.isnan(ym):
                        fcontent.append([VARNAME[var],
                                         '%d' % YEAR[row],
                                         '%d' % MONTH[row],
                                         '%d' % DAY[row],
                                         '%s' % log_Ndat[row, var],
                                         '%0.2f' % log_RMSE[row, var],
                                         '%0.1f' % (yp - ym),
                                         '%0.1f' % yp,
                                         '%0.1f' % ym])
                        fcontent[-1].extend(xm)

            # ---- Save File ----

            fname = '%s (%s)_%s-%s.err' % (clean_tarStaName,
                                           target_station_clim,
                                           YearStart, YearEnd)

            output_path = os.path.join(dirname, fname)
            self.save_content_to_file(output_path, fcontent)
            print('Generating %s.' % fname)

            # ---- Plot some graphs ----

            pperr = PostProcessErr(output_path)
            pperr.set_fig_format(self.fig_format)
            pperr.set_fig_language(self.fig_language)
            pperr.generates_graphs()

            # ---- SOME CALCULATIONS ----

            RMSE = np.zeros(nVAR)
            ERRMAX = np.zeros(nVAR)
            ERRSUM = np.zeros(nVAR)
            for i in range(nVAR):

                errors = YpFULL[:, i] - Y2fill[:, i]
                errors = errors[~np.isnan(errors)]

                rmse = errors**2
                rmse = rmse[rmse != 0]
                rmse = np.mean(rmse)**0.5

                errmax = np.abs(errors)
                errmax = np.max(errmax)

                errsum = np.sum(errors)

                RMSE[i] = rmse
                ERRMAX[i] = errmax
                ERRSUM[i] = errsum

            print('RMSE :')
            print(np.round(RMSE, 2))
            print('Maximum Error :')
            print(ERRMAX)
            print('Cumulative Error :')
            print(ERRSUM)

        self.STOP = False
        self.sig_gapfill_finished.emit(True)

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

    def sort_sta_corrcoef(self, var):
        corcoef = self.corcoef[var]
        corcoef = corcoef.sort_values(ascending=False)
        corcoef = corcoef[corcoef.notnull()]

        sorted_stations = corcoef.index.tolist()
        sorted_stations.remove(self.target)
        sorted_stations.insert(0, self.target)
        # Note that we need to move the target station explicitely at the
        # start of the list in case there is another station with data
        # duplicated from the target station.

        return sorted_stations

    @staticmethod
    def save_content_to_file(fname, fcontent):
        """Save content to a coma-separated value text file."""
        save_content_to_csv(fname, fcontent)

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
        save_content_to_csv(output_path, fcontent)

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


def main():                                                  # pragma: no cover

    # =========================================================================
    # 1 - Create an instance of the class *GapFillWeather*
    # =========================================================================
    # The algorithm is built as a base class of the Qt GUI Framework
    # using the PyQt binding. Signals are also emitted at various stade
    # in the gap-filling routine. This has been done to facilitate the
    # addition of a Graphical User Interface on top of the algorithm with
    # the Qt GUI Development framework.
    gapfill_weather = GapFillWeather()

    # =========================================================================
    # 2 - Setup input and output directory
    # =========================================================================

    # Weather data files must be put all together in the input directory.
    # The outputs produced by the algorithm after a gap-less weather dataset
    # was produced for the target station will be saved within the output
    # directory, in a sub-folder named after the name of the target station.

    gapfill_weather.inputDir = 'C:/Users/User/pygwd/pygwd/tests/data'
    gapfill_weather.outputDir = 'C:/Users/User/pygwd/pygwd/tests/data/Output'

    # =========================================================================
    # 3 - Load weather the data files
    # =========================================================================
    # Datafiles are loaded directly from the input directory defined in
    # step 2.
    stanames = gapfill_weather.load_data()
    print(stanames)

    # =========================================================================
    # 4 - Setup target station
    # =========================================================================
    gapfill_weather.set_target_station('7023270')

    # =========================================================================
    # 5 - Define the time plage
    # =========================================================================
    # Gaps in the weather data will be filled only between *time_start* and
    # *time_end*
    gapfill_weather.time_start = gapfill_weather.WEATHER.datetimes[0]
    gapfill_weather.time_end = gapfill_weather.WEATHER.datetimes[-1]

    # =========================================================================
    # 6 - Setup method parameters
    # =========================================================================
    # See the help of class *GapFillWeather* for a description of each
    # parameter.
    gapfill_weather.NSTAmax = 3
    gapfill_weather.limitDist = 100
    gapfill_weather.limitAlt = 350
    gapfill_weather.full_error_analysis = False
    gapfill_weather.leave_one_out = False
    gapfill_weather.regression_mode = 0
    # 0 -> Least Absolute Deviation (LAD)
    # 1 -> Ordinary Least-Square (OLS)

    # =========================================================================
    # 7 - Gap-fill the data of the target station
    # =========================================================================
    # A gap-less weather dataset will be produced for the target weather
    # station defined in step 4, for the time plage defined in step 5.

    # To run the algorithm in batch mode, simply loop over all the indexes of
    # the list *staname* where the target station is redefined at each
    # iteration as in step 4 and rerun the *fill_data* method each time.

    gapfill_weather.fill_data()


if __name__ == '__main__':
    main()
