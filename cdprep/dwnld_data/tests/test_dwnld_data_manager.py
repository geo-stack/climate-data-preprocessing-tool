# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

"""
Tests for the WeatherStationDownloader.
"""

# ---- Standard imports
from datetime import datetime
import os
import os.path as osp
os.environ['CDPREP_PYTEST'] = 'True'

# ---- Third party imports
import pytest

# ---- Local imports
from cdprep.config.main import CONF
from cdprep.dwnld_data.weather_station_finder import DATABASE_FILEPATH
from cdprep.dwnld_data.dwnld_data_manager import WeatherStationDownloader


# =============================================================================
# ---- Fixtures
# =============================================================================
@pytest.fixture(scope='module')
def workdir(tmp_path_factory):
    """A fixture for a test working directory."""
    workdir = osp.join(
        tmp_path_factory.getbasetemp(), "@ tèst-fïl! 'dätèt!")
    os.makedirs(workdir, exist_ok=True)
    return workdir


@pytest.fixture()
def dwnld_manager(qtbot, mocker, workdir):
    """A fixture for the WeatherDataGapfiller."""
    # We need to reset the configs to defaults after each test to make sure
    # they can be run independently one from another.
    CONF.reset_to_defaults()

    dwnld_manager = WeatherStationDownloader()
    qtbot.addWidget(dwnld_manager)
    assert dwnld_manager._database_isloading is True
    qtbot.waitUntil(lambda: dwnld_manager._database_isloading is False,
                    timeout=5000)
    assert osp.exists(DATABASE_FILEPATH)

    # Check that the number of stations displayed in the table is as expected.
    nstations = len(dwnld_manager.stn_finder_worker._data)
    assert len(dwnld_manager.station_table.get_stationlist()) == nstations
    assert len(dwnld_manager.stn_finder_worker.get_stationlist()) == nstations

    # Check default falues for the proximity filter.
    assert dwnld_manager.prox_grpbox.isChecked() is False
    assert dwnld_manager.lat_spinBox.value() == 0
    assert dwnld_manager.lon_spinBox.value() == 0
    assert dwnld_manager.radius_SpinBox.currentText() == '25 km'

    # Check default falues for the data availability filter.
    assert dwnld_manager.year_widg.isChecked() is False
    assert dwnld_manager.nbrYear.value() == 3
    assert dwnld_manager.minYear.value() == 1840
    assert dwnld_manager.maxYear.value() == datetime.now().year

    return dwnld_manager


# =============================================================================
# ---- Tests for the WeatherStationDownloader
# =============================================================================
def test_browse_station(dwnld_manager, workdir, qtbot):
    """
    Test that browsing climate stations in the WeatherStationDownloader is
    working as expected.
    """
    # Check that the proximity filter is working as expected.
    dwnld_manager.lat_spinBox.setValue(45)
    dwnld_manager.lon_spinBox.setValue(74)
    dwnld_manager.radius_SpinBox.setCurrentIndex(1)
    dwnld_manager.prox_grpbox.setChecked(True)
    assert len(dwnld_manager.station_table.get_stationlist()) == 31

    # Check that the data availability filter is working as expected.
    dwnld_manager.nbrYear.setValue(10)
    dwnld_manager.minYear.setValue(2000)
    dwnld_manager.maxYear.setValue(2020)
    dwnld_manager.year_widg.setChecked(True)
    assert len(dwnld_manager.station_table.get_stationlist()) == 11

    # Check that the error that was reported in Issue#17 is not
    # triggered anymore.
    # See cgq-qgc/climate-data-preprocessing-tool#17.
    dwnld_manager.lat_spinBox.setValue(4)
    assert len(dwnld_manager.station_table.get_stationlist()) == 0


if __name__ == "__main__":
    pytest.main(['-x', __file__, '-v', '-rw'])
