# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

# ---- Standard imports
from time import sleep
import os
import os.path as osp

# ---- Third party imports
from PyQt5.QtCore import pyqtSlot as QSlot
from PyQt5.QtCore import pyqtSignal as QSignal
from PyQt5.QtCore import Qt, QThread, QDate
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QGridLayout, QFrame, QLabel, QComboBox,
    QTextEdit, QDateEdit, QSpinBox, QRadioButton, QCheckBox, QProgressBar,
    QApplication, QMessageBox, QToolButton, QTabWidget, QGroupBox)

# ---- Local imports
from cdprep.config.main import CONF
from cdprep.config.icons import get_icon, get_iconsize
from cdprep.gapfill_data.gapfill_weather_algorithm import DataGapfiller
from cdprep.utils.ospath import delete_file
from cdprep.utils.qthelpers import datetime_from_qdatedit


class WeatherDataGapfiller(QWidget):

    ConsoleSignal = QSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.isFillAll_inProgress = False

        # Correlation calculation won't be triggered by events when
        # CORRFLAG is 'off'
        self.CORRFLAG = 'on'

        self.__initUI__()

        # Setup gap fill worker and thread :
        self.gapfill_worker = DataGapfiller()
        self.gapfill_worker.sig_gapfill_finished.connect(
            self.gapfill_worker_return)
        self.gapfill_worker.sig_gapfill_progress.connect(
            self.progressbar.setValue)
        self.gapfill_worker.sig_console_message.connect(
            self.ConsoleSignal.emit)

        self.gapfill_thread = QThread()
        self.gapfill_worker.moveToThread(self.gapfill_thread)
        self.gapfill_thread.started.connect(
            self.gapfill_worker.gapfill_data)

    def __initUI__(self):
        self.setWindowIcon(get_icon('master'))

        # Setup the toolbar at the bottom.
        self.btn_fill = QPushButton('Gapfill Data')
        self.btn_fill.setIcon(get_icon('fill_data'))
        self.btn_fill.setIconSize(get_iconsize('small'))
        self.btn_fill.setToolTip(
            "Fill the gaps in the daily weather data of the selected "
            "weather station.")
        self.btn_fill.clicked.connect(self._handle_gapfill_btn_clicked)

        widget_toolbar = QFrame()
        grid_toolbar = QGridLayout(widget_toolbar)
        grid_toolbar.addWidget(self.btn_fill, 0, 1)
        grid_toolbar.setContentsMargins(0, 0, 0, 0)
        grid_toolbar.setColumnStretch(0, 100)

        # ---- Target Station groupbox
        self.target_station = QComboBox()
        self.target_station.currentIndexChanged.connect(
            self._handle_target_station_changed)

        self.target_station_info = QTextEdit()
        self.target_station_info.setReadOnly(True)
        self.target_station_info.setMaximumHeight(110)

        self.btn_refresh_staList = QToolButton()
        self.btn_refresh_staList.setIcon(get_icon('refresh'))
        self.btn_refresh_staList.setToolTip(
            'Force the reloading of the weather data files')
        self.btn_refresh_staList.setIconSize(get_iconsize('small'))
        self.btn_refresh_staList.setAutoRaise(True)
        self.btn_refresh_staList.clicked.connect(self.btn_refresh_isclicked)

        self.btn_delete_data = QToolButton()
        self.btn_delete_data.setIcon(get_icon('delete_data'))
        self.btn_delete_data.setEnabled(False)
        self.btn_delete_data.setAutoRaise(True)
        self.btn_delete_data.setToolTip(
            'Remove the currently selected dataset and delete the input '
            'datafile. However, raw datafiles will be kept.')
        self.btn_delete_data.clicked.connect(self.delete_current_dataset)

        # Generate the layout for the target station group widget.
        self.target_widget = QWidget()
        target_station_layout = QGridLayout(self.target_widget)
        target_station_layout.setHorizontalSpacing(1)
        target_station_layout.setColumnStretch(0, 1)
        target_station_layout.setContentsMargins(0, 0, 0, 0)

        widgets = [self.target_station, self.btn_refresh_staList,
                   self.btn_delete_data]
        target_station_layout.addWidget(self.target_station, 1, 0)
        for col, widget in enumerate(widgets):
            target_station_layout.addWidget(widget, 1, col)

        # Setup the gapfill dates.
        label_From = QLabel('From :  ')
        self.date_start_widget = QDateEdit()
        self.date_start_widget.setDisplayFormat('dd / MM / yyyy')
        self.date_start_widget.setEnabled(False)
        self.date_start_widget.dateChanged.connect(
            self.correlation_table_display)

        label_To = QLabel('To :  ')
        self.date_end_widget = QDateEdit()
        self.date_end_widget.setEnabled(False)
        self.date_end_widget.setDisplayFormat('dd / MM / yyyy')
        self.date_end_widget.dateChanged.connect(
            self.correlation_table_display)

        self.fillDates_widg = QWidget()
        gapfilldates_layout = QGridLayout(self.fillDates_widg)
        gapfilldates_layout.addWidget(label_From, 0, 0)
        gapfilldates_layout.addWidget(self.date_start_widget, 0, 1)
        gapfilldates_layout.addWidget(label_To, 1, 0)
        gapfilldates_layout.addWidget(self.date_end_widget, 1, 1)
        gapfilldates_layout.setColumnStretch(2, 1)
        gapfilldates_layout.setContentsMargins(0, 0, 0, 0)

        # Create the gapfill target station groupbox.
        target_groupbox = QGroupBox("Fill data for weather station")
        target_layout = QGridLayout(target_groupbox)
        target_layout.addWidget(self.target_widget, 0, 0)
        target_layout.addWidget(self.target_station_info, 1, 0)
        target_layout.addWidget(self.fillDates_widg, 2, 0)

        # Setup the left panel.
        self._regression_model_groupbox = (
            self._create_regression_model_settings())
        self._station_selection_groupbox = (
            self._create_station_selection_criteria())

        self.left_panel = QFrame()
        left_panel_layout = QGridLayout(self.left_panel)
        left_panel_layout.addWidget(target_groupbox, 0, 0)
        left_panel_layout.addWidget(self._station_selection_groupbox, 3, 0)
        left_panel_layout.addWidget(self._regression_model_groupbox, 4, 0)
        left_panel_layout.addWidget(widget_toolbar, 5, 0)
        left_panel_layout.setRowStretch(6, 1)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)

        # Setup the right panel.
        self.corrcoeff_textedit = QTextEdit()
        self.corrcoeff_textedit.setReadOnly(True)
        self.corrcoeff_textedit.setMinimumWidth(700)
        self.corrcoeff_textedit.setFrameStyle(0)
        self.corrcoeff_textedit.document().setDocumentMargin(10)

        self.sta_display_summary = QTextEdit()
        self.sta_display_summary.setReadOnly(True)
        self.sta_display_summary.setFrameStyle(0)
        self.sta_display_summary.document().setDocumentMargin(10)

        right_panel = QTabWidget()
        right_panel.addTab(self.corrcoeff_textedit, 'Correlation Coefficients')
        right_panel.addTab(self.sta_display_summary, 'Data Overview')

        # Setup the progressbar.
        self.progressbar = QProgressBar()
        self.progressbar.setValue(0)
        self.progressbar.hide()

        # Setup the main grid.
        main_grid = QGridLayout(self)
        main_grid.addWidget(self.left_panel, 0, 0)
        main_grid.addWidget(right_panel, 0, 1)
        main_grid.addWidget(self.progressbar, 1, 0, 1, 2)
        main_grid.setColumnStretch(1, 500)
        main_grid.setRowStretch(0, 500)

    def _create_station_selection_criteria(self):
        Nmax_label = QLabel('Nbr. of stations :')
        self.Nmax = QSpinBox()
        self.Nmax.setRange(0, 99)
        self.Nmax.setValue(CONF.get('gapfill_data', 'nbr_of_station', 4))
        self.Nmax.setAlignment(Qt.AlignCenter)

        ttip = ('<p>Distance limit beyond which neighboring stations'
                ' are excluded from the gapfilling procedure.</p>'
                '<p>This condition is ignored if set to -1.</p>')
        distlimit_label = QLabel('Max. Distance :')
        distlimit_label.setToolTip(ttip)
        self.distlimit = QSpinBox()
        self.distlimit.setRange(-1, 9999)
        self.distlimit.setSingleStep(1)
        self.distlimit.setValue(
            CONF.get('gapfill_data', 'max_horiz_dist', 100))
        self.distlimit.setToolTip(ttip)
        self.distlimit.setSuffix(' km')
        self.distlimit.setAlignment(Qt.AlignCenter)
        self.distlimit.valueChanged.connect(self.correlation_table_display)

        ttip = ('<p>Altitude difference limit over which neighboring '
                ' stations are excluded from the gapfilling procedure.</p>'
                '<p>This condition is ignored if set to -1.</p>')
        altlimit_label = QLabel('Max. Elevation Diff. :')
        altlimit_label.setToolTip(ttip)
        self.altlimit = QSpinBox()
        self.altlimit.setRange(-1, 9999)
        self.altlimit.setSingleStep(1)
        self.altlimit.setValue(
            CONF.get('gapfill_data', 'max_vert_dist', 350))
        self.altlimit.setToolTip(ttip)
        self.altlimit.setSuffix(' m')
        self.altlimit.setAlignment(Qt.AlignCenter)
        self.altlimit.valueChanged.connect(self.correlation_table_display)

        # Setup the main widget.
        widget = QGroupBox('Stations Selection Criteria')
        layout = QGridLayout(widget)

        layout.addWidget(Nmax_label, 0, 0)
        layout.addWidget(self.Nmax, 0, 1)
        layout.addWidget(distlimit_label, 1, 0)
        layout.addWidget(self.distlimit, 1, 1)
        layout.addWidget(altlimit_label, 2, 0)
        layout.addWidget(self.altlimit, 2, 1)
        layout.setColumnStretch(0, 1)

        return widget

    def _create_advanced_settings(self):
        self.full_error_analysis = QCheckBox('Full Error Analysis.')
        self.full_error_analysis.setChecked(True)

        fig_opt_layout = QGridLayout()
        fig_opt_layout.addWidget(QLabel("Figure output format : "), 0, 0)
        fig_opt_layout.addWidget(self.fig_format, 0, 2)
        fig_opt_layout.addWidget(QLabel("Figure labels language : "), 1, 0)
        fig_opt_layout.addWidget(self.fig_language, 1, 2)

        fig_opt_layout.setContentsMargins(0, 0, 0, 0)
        fig_opt_layout.setColumnStretch(1, 100)

        # Setup the main layout.
        widget = QFrame()
        layout = QGridLayout(widget)
        layout.addWidget(self.full_error_analysis, 0, 0)
        layout.addLayout(fig_opt_layout, 2, 0)
        layout.setRowStretch(layout.rowCount(), 100)
        layout.setContentsMargins(10, 0, 10, 0)

        return widget

    def _create_regression_model_settings(self):
        self.RMSE_regression = QRadioButton('Ordinary Least Squares')
        self.RMSE_regression.setChecked(
            CONF.get('gapfill_data', 'regression_model', 'OLS') == 'OLS')

        self.ABS_regression = QRadioButton('Least Absolute Deviations')
        self.ABS_regression.setChecked(
            CONF.get('gapfill_data', 'regression_model', 'OLS') == 'LAD')

        widget = QGroupBox('Regression Model')
        layout = QGridLayout(widget)
        layout.addWidget(self.RMSE_regression, 0, 0)
        layout.addWidget(self.ABS_regression, 1, 0)

        return widget

    @property
    def workdir(self):
        return self.__workdir

    def set_workdir(self, dirname):
        """
        Set the working directory to dirname.
        """
        self.__workdir = dirname
        self.gapfill_worker.inputDir = dirname
        self.load_data_dir_content()

    def delete_current_dataset(self):
        """
        Delete the current dataset source file and force a reload of the input
        daily weather datafiles.
        """
        current_index = self.target_station.currentIndex()
        if current_index != -1:
            basename = self.gapfill_worker.WEATHER.fnames[current_index]
            dirname = self.gapfill_worker.inputDir
            filename = os.path.join(dirname, basename)
            delete_file(filename)
            self.load_data_dir_content()

    def btn_refresh_isclicked(self):
        """
        Handles when the button to refresh the list of input daily weather
        datafiles is clicked
        """
        self.load_data_dir_content()

    def load_data_dir_content(self):
        """
        Initiate the loading of weater data files contained in the
        */Meteo/Input folder and display the resulting station list in the
        target station combobox.
        """
        # Reset the GUI.
        self.corrcoeff_textedit.setText('')
        self.target_station_info.setText('')
        self.target_station.clear()
        QApplication.processEvents()

        # Load data and fill UI with info.
        self.CORRFLAG = 'off'
        self.gapfill_worker.load_data()
        station_names = self.gapfill_worker.wxdatasets.station_names
        station_ids = self.gapfill_worker.wxdatasets.station_ids
        for station_name, station_id in zip(station_names, station_ids):
            self.target_station.addItem(station_name, userData=station_id)
        self.sta_display_summary.setHtml(
            self.gapfill_worker.generate_html_summary_table())

        if len(station_names) > 0:
            self.set_fill_and_save_dates()
            self.target_station.blockSignals(True)
            self.target_station.setCurrentIndex(0)
            self.target_station.blockSignals(False)
        self.CORRFLAG = 'on'
        self._handle_target_station_changed(self.target_station.currentIndex())

    def set_fill_and_save_dates(self):
        """
        Set first and last dates of the data serie in the boxes of the
        *Fill and Save* area.
        """
        if self.gapfill_worker.wxdatasets.count():
            self.date_start_widget.setEnabled(True)
            self.date_end_widget.setEnabled(True)

            mindate = (
                self.gapfill_worker.wxdatasets.metadata['first_date'].min())
            maxdate = (
                self.gapfill_worker.wxdatasets.metadata['last_date'].max())
            qdatemin = QDate(mindate.year, mindate.month, mindate.day)
            qdatemax = QDate(maxdate.year, maxdate.month, maxdate.day)

            self.date_start_widget.setDate(qdatemin)
            self.date_start_widget.setMinimumDate(qdatemin)
            self.date_start_widget.setMaximumDate(qdatemax)

            self.date_end_widget.setDate(qdatemax)
            self.date_end_widget.setMinimumDate(qdatemin)
            self.date_end_widget.setMaximumDate(qdatemax)

    def correlation_table_display(self):
        """
        This method plot the table in the display area.

        It is separated from the method <update_corrcoeff> because red
        numbers and statistics regarding missing data for the selected
        time period can be updated in the table when the user changes the
        values without having to recalculate the correlation coefficient
        each time.
        """
        if self.CORRFLAG == 'off' or self.target_station.currentIndex() == -1:
            return
        table, target_info = (
            self.gapfill_worker.generate_correlation_html_table(
                self.get_gapfill_parameters()))
        self.corrcoeff_textedit.setText(table)
        self.target_station_info.setText(target_info)

    @QSlot(int)
    def _handle_target_station_changed(self, index):
        """Handle when the target station is changed by the user."""
        self.btn_delete_data.setEnabled(index != -1)
        if index != -1:
            self.update_corrcoeff()

    def update_corrcoeff(self):
        """
        Calculate the correlation coefficients and display the results
        in the GUI.
        """
        if self.CORRFLAG == 'on' and self.target_station.currentIndex() != -1:
            station_id = self.target_station.currentData()
            self.gapfill_worker.set_target_station(station_id)
            print("Correlation coefficients calculated for station {}.".format(
                self.gapfill_worker.get_target_station()['Station Name']))
            self.correlation_table_display()

    def restore_gui(self):
        self.btn_fill.setIcon(get_icon('fill_data'))
        self.btn_fill.setEnabled(True)

        self.target_widget.setEnabled(True)
        self.fillDates_widg.setEnabled(True)
        self._regression_model_groupbox.setEnabled(False)
        self._station_selection_groupbox.setEnabled(False)
        self.progressbar.setValue(0)
        QApplication.processEvents()
        self.progressbar.hide()

    def get_gapfill_parameters(self):
        """
        Return a dictionary containing the parameters that are set in the GUI
        for gapfilling weather data.
        """
        return {
            'limitDist': self.distlimit.value(),
            'limitAlt': self.altlimit.value(),
            'date_start': self.date_start_widget.date().toString('dd/MM/yyyy'),
            'date_end': self.date_end_widget.date().toString('dd/MM/yyyy')
            }

    def get_dataset_names(self):
        """
        Return a list of the names of the dataset that are loaded in
        memory and listed in the target station dropdown menu.
        """
        return [self.target_station.itemText(i) for i in
                range(self.target_station.count())]

    def _handle_gapfill_btn_clicked(self):
        """
        Handle when the user clicked on the gapfill button.
        """
        if self.gapfill_worker.wxdatasets.count() == 0:
            QMessageBox.warning(
                self, 'Warning', "There is no data to fill.", QMessageBox.Ok)
            return

        # Check for dates errors.
        datetime_start = datetime_from_qdatedit(self.date_start_widget)
        datetime_end = datetime_from_qdatedit(self.date_end_widget)
        if datetime_start > datetime_end:
            QMessageBox.warning(
                self, 'Warning',
                ("<i>From</i> date is set to a later time than "
                 "the <i>To</i> date."),
                QMessageBox.Ok)
            return
        if self.target_station.currentIndex() == -1:
            QMessageBox.warning(
                self, 'Warning',
                "No weather station is currently selected",
                QMessageBox.Ok)
            return

        # Disable GUI and continue the process normally
        self.btn_fill.setEnabled(False)
        self.fillDates_widg.setEnabled(False)
        self.target_widget.setEnabled(False)
        self._regression_model_groupbox.setEnabled(False)
        self._station_selection_groupbox.setEnabled(False)
        self.progressbar.show()

        self.isFillAll_inProgress = False
        sta_indx2fill = self.target_station.currentIndex()
        self.gap_fill_start(sta_indx2fill)

    def gapfill_worker_return(self, event):
        """
        Method initiated from an automatic return from the gapfilling
        process in batch mode. Iterate over the station list and continue
        process normally.
        """
        self.gapfill_thread.quit()
        if event:
            sta_indx2fill = self.target_station.currentIndex() + 1
            if (self.isFillAll_inProgress is False or
                    sta_indx2fill == self.gapfill_worker.wxdatasets.count()):
                # Single fill process completed sucessfully for the current
                # selected weather station OR Fill All process completed
                # sucessfully for all the weather stations in the list.
                self.isFillAll_inProgress = False
                self.restore_gui()
            else:
                self.gap_fill_start(sta_indx2fill)
        else:
            print('Gap-filling routine stopped.')
            # The gapfilling routine was stopped from the UI.
            self.isFillAll_inProgress = False
            self.restore_gui()

    def gap_fill_start(self, sta_indx2fill):
        # Wait for the QThread to finish.
        waittime = 0
        while self.gapfill_thread.isRunning():
            print('Waiting for the fill weather data thread to close ' +
                  'before processing with the next station.')
            sleep(0.1)
            waittime += 0.1
            if waittime > 15:
                msg = ('This function is not working as intended.' +
                       ' Please report a bug.')
                print(msg)
                self.ConsoleSignal.emit('<font color=red>%s</font>' % msg)
                return

        # Update the GUI.
        self.CORRFLAG = 'off'
        self.target_station.setCurrentIndex(sta_indx2fill)
        self.CORRFLAG = 'on'

        # Calculate correlation coefficient for the next station.
        self.update_corrcoeff()

        # Start the gapfill thread.
        self.gapfill_worker.time_start = datetime_from_qdatedit(
            self.date_start_widget)
        self.gapfill_worker.time_end = datetime_from_qdatedit(
            self.date_end_widget)
        self.gapfill_worker.NSTAmax = self.Nmax.value()
        self.gapfill_worker.limitDist = self.distlimit.value()
        self.gapfill_worker.limitAlt = self.altlimit.value()
        self.gapfill_worker.regression_mode = self.RMSE_regression.isChecked()

        self.gapfill_thread.start()

    def close(self):
        CONF.set('gapfill_data', 'nbr_of_station', self.Nmax.value())
        CONF.set('gapfill_data', 'max_horiz_dist', self.distlimit.value())
        CONF.set('gapfill_data', 'max_vert_dist', self.altlimit.value())
        CONF.set('gapfill_data', 'regression_model',
                 'OLS' if self.RMSE_regression.isChecked() else 'LAD')
        super().close()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    window = WeatherDataGapfiller()
    window.show()
    # w.set_workdir("C:\\Users\\jsgosselin\\GWHAT\\Projects\\Example")
    # w.load_data_dir_content()

    # lat = w.gapfill_worker.WEATHER.LAT
    # lon = w.gapfill_worker.WEATHER.LON
    # name = w.gapfill_worker.WEATHER.STANAME
    # alt = w.gapfill_worker.WEATHER.ALT

    sys.exit(app.exec_())
