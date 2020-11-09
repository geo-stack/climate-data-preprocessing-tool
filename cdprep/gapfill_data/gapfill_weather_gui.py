# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

# ---- Standard imports
import os
import os.path as osp

# ---- Third party imports
from PyQt5.QtCore import pyqtSlot as QSlot
from PyQt5.QtCore import pyqtSignal as QSignal
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QGridLayout, QFrame, QLabel, QComboBox,
    QTextEdit, QDateEdit, QSpinBox, QRadioButton, QCheckBox, QProgressBar,
    QApplication, QMessageBox, QToolButton, QTabWidget, QGroupBox,
    QMainWindow)

# ---- Local imports
from cdprep.config.main import CONF
from cdprep.config.icons import get_icon, get_iconsize
from cdprep.gapfill_data.gapfill_weather_algorithm import DataGapfillManager
from cdprep.utils.ospath import delete_file
from cdprep.utils.qthelpers import datetime_from_qdatedit


class WeatherDataGapfiller(QMainWindow):

    ConsoleSignal = QSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workdir = None

        self._corrcoeff_update_inprogress = False
        self._pending_corrcoeff_update = None
        self._loading_data_inprogress = False

        self.__initUI__()

        # Setup the DataGapfillManager.
        self.gapfill_manager = DataGapfillManager()
        self.gapfill_manager.sig_task_progress.connect(
            self.progressbar.setValue)
        self.gapfill_manager.sig_status_message.connect(
            self.set_statusbar_text)

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
        self.btn_refresh_staList.clicked.connect(
            lambda: self.load_data_dir_content(force_reload=True))

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
            self._update_corrcoeff_table)

        label_To = QLabel('To :  ')
        self.date_end_widget = QDateEdit()
        self.date_end_widget.setEnabled(False)
        self.date_end_widget.setDisplayFormat('dd / MM / yyyy')
        self.date_end_widget.dateChanged.connect(
            self._update_corrcoeff_table)

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

        self.right_panel = QTabWidget()
        self.right_panel.addTab(
            self.corrcoeff_textedit, 'Correlation Coefficients')
        self.right_panel.addTab(
            self.sta_display_summary, 'Data Overview')

        # Setup the progressbar.
        self.progressbar = QProgressBar()
        self.progressbar.setValue(0)
        self.progressbar.hide()

        self.statustext = QLabel()
        self.statustext.setStyleSheet(
            "QLabel {background-color: transparent; padding: 0 0 0 3px;}")
        self.statustext.setMinimumHeight(self.progressbar.minimumHeight())

        # Setup the main widget.
        main_widget = QWidget()
        main_grid = QGridLayout(main_widget)
        main_grid.addWidget(self.left_panel, 0, 0)
        main_grid.addWidget(self.right_panel, 0, 1)
        main_grid.addWidget(self.progressbar, 1, 0, 1, 2)
        main_grid.addWidget(self.statustext, 1, 0, 1, 2)
        main_grid.setColumnStretch(1, 500)
        main_grid.setRowStretch(0, 500)
        self.setCentralWidget(main_widget)

    def _create_station_selection_criteria(self):
        Nmax_label = QLabel('Nbr. of stations :')
        self.Nmax = QSpinBox()
        self.Nmax.setRange(0, 99)
        self.Nmax.setMinimum(1)
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
        self.distlimit.valueChanged.connect(self._update_corrcoeff_table)

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
        self.altlimit.valueChanged.connect(self._update_corrcoeff_table)

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

    def set_statusbar_text(self, text):
        self.statustext.setText(text)

    @property
    def workdir(self):
        return self._workdir

    def set_workdir(self, dirname):
        """
        Set the working directory to dirname.
        """
        self._workdir = dirname
        self.gapfill_manager.set_workdir(dirname)
        self.load_data_dir_content()

    def delete_current_dataset(self):
        """
        Delete the current dataset source file and force a reload of the input
        daily weather datafiles.
        """
        current_index = self.target_station.currentIndex()
        if current_index != -1:
            basename = self.gapfill_manager.worker().wxdatasets.fnames[
                current_index]
            filename = os.path.join(self.workdir, basename)
            delete_file(filename)
            self.load_data_dir_content()

    def _handle_target_station_changed(self):
        """Handle when the target station is changed by the user."""
        self.btn_delete_data.setEnabled(
            self.target_station.currentIndex() != -1)
        self.update_corrcoeff()

    def get_dataset_names(self):
        """
        Return a list of the names of the dataset that are loaded in
        memory and listed in the target station dropdown menu.
        """
        return [self.target_station.itemText(i) for i in
                range(self.target_station.count())]

    # ---- Correlation coefficients
    def update_corrcoeff(self):
        """
        Calculate the correlation coefficients and display the results
        in the GUI.
        """
        if self.target_station.currentIndex() != -1:
            station_id = self.target_station.currentData()
            if self._corrcoeff_update_inprogress is True:
                self._pending_corrcoeff_update = station_id
            else:
                self._corrcoeff_update_inprogress = True
                self.gapfill_manager.set_target_station(
                    station_id, callback=self._handle_corrcoeff_updated)

    def _handle_corrcoeff_updated(self):
        self._corrcoeff_update_inprogress = False
        if self._pending_corrcoeff_update is None:
            self._update_corrcoeff_table()
        else:
            self._pending_corrcoeff_update = None
            self.update_corrcoeff()

    def _update_corrcoeff_table(self):
        """
        This method plot the correlation coefficient table in the display area.

        It is separated from the method "update_corrcoeff" because red
        numbers and statistics regarding missing data for the selected
        time period can be updated in the table when the user changes the
        values without having to recalculate the correlation coefficient
        each time.
        """
        if self.target_station.currentIndex() != -1:
            table, target_info = (
                self.gapfill_manager.worker().generate_correlation_html_table(
                    self.get_gapfill_parameters()))
            self.corrcoeff_textedit.setText(table)
            self.target_station_info.setText(target_info)

    # ---- Load Data
    def load_data_dir_content(self, force_reload=False):
        """
        Load weater data from valid files contained in the working directory.
        """
        self._pending_corrcoeff_update = None
        self._loading_data_inprogress = True
        self.left_panel.setEnabled(False)
        self.right_panel.setEnabled(False)

        self.corrcoeff_textedit.setText('')
        self.target_station_info.setText('')
        self.target_station.clear()

        self.gapfill_manager.load_data(
            force_reload=force_reload,
            callback=self._handle_data_dir_content_loaded)

    def _handle_data_dir_content_loaded(self):
        """
        Handle when data finished loaded from valid files contained in
        the working directory.
        """
        self.left_panel.setEnabled(True)
        self.right_panel.setEnabled(True)

        self.target_station.blockSignals(True)
        station_names = self.gapfill_manager.get_station_names()
        station_ids = self.gapfill_manager.get_station_ids()
        for station_name, station_id in zip(station_names, station_ids):
            self.target_station.addItem(station_name, userData=station_id)
        self.target_station.blockSignals(False)

        self.sta_display_summary.setHtml(
            self.gapfill_manager.worker().generate_html_summary_table())

        if len(station_names) > 0:
            self._setup_fill_and_save_dates()
            self.target_station.blockSignals(True)
            self.target_station.setCurrentIndex(0)
            self.target_station.blockSignals(False)
        self._handle_target_station_changed()
        self._loading_data_inprogress = False

    def _setup_fill_and_save_dates(self):
        """
        Set first and last dates of the 'Fill data for weather station'.
        """
        if self.gapfill_manager.count():
            self.date_start_widget.setEnabled(True)
            self.date_end_widget.setEnabled(True)

            mindate = (
                self.gapfill_manager.worker()
                .wxdatasets.metadata['first_date'].min())
            maxdate = (
                self.gapfill_manager.worker()
                .wxdatasets.metadata['last_date'].max())

            qdatemin = QDate(mindate.year, mindate.month, mindate.day)
            qdatemax = QDate(maxdate.year, maxdate.month, maxdate.day)

            self.date_start_widget.blockSignals(True)
            self.date_start_widget.setDate(qdatemin)
            self.date_start_widget.setMinimumDate(qdatemin)
            self.date_start_widget.setMaximumDate(qdatemax)
            self.date_start_widget.blockSignals(False)

            self.date_end_widget.blockSignals(True)
            self.date_end_widget.setDate(qdatemax)
            self.date_end_widget.setMinimumDate(qdatemin)
            self.date_end_widget.setMaximumDate(qdatemax)
            self.date_end_widget.blockSignals(False)

    # ---- Gapfill Data
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

    def _handle_gapfill_btn_clicked(self):
        """
        Handle when the user clicked on the gapfill button.
        """
        if self.gapfill_manager.count() == 0:
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

        self.start_gapfill_target()

    def _handle_gapfill_target_finished(self):
        """
        Method initiated from an automatic return from the gapfilling
        process in batch mode. Iterate over the station list and continue
        process normally.
        """
        self.btn_fill.setIcon(get_icon('fill_data'))
        self.btn_fill.setEnabled(True)

        self.target_widget.setEnabled(True)
        self.fillDates_widg.setEnabled(True)
        self._regression_model_groupbox.setEnabled(True)
        self._station_selection_groupbox.setEnabled(True)
        self.progressbar.setValue(0)
        QApplication.processEvents()
        self.progressbar.hide()

    def start_gapfill_target(self):
        # Update the gui.
        self.btn_fill.setEnabled(False)
        self.fillDates_widg.setEnabled(False)
        self.target_widget.setEnabled(False)
        self._regression_model_groupbox.setEnabled(False)
        self._station_selection_groupbox.setEnabled(False)
        self.progressbar.show()

        # Start the gapfill thread.
        self.gapfill_manager.gapfill_data(
            time_start=datetime_from_qdatedit(self.date_start_widget),
            time_end=datetime_from_qdatedit(self.date_end_widget),
            max_neighbors=self.Nmax.value(),
            hdist_limit=self.distlimit.value(),
            vdist_limit=self.altlimit.value(),
            regression_mode=self.RMSE_regression.isChecked(),
            callback=self._handle_gapfill_target_finished
            )

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
