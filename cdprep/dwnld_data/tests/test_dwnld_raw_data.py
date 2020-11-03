# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

"""
Tests for the RawDataDownloader.
"""

# ---- Standard imports
import os
import os.path as osp
os.environ['CDPREP_PYTEST'] = 'True'

# ---- Third party imports
import pytest

# ---- Local imports
from cdprep.dwnld_data.dwnld_data_manager import (
    RawDataDownloader, read_raw_datafile)


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
def dwnld_worker(qtbot, mocker, workdir):
    """A fixture for the WeatherDataGapfiller."""
    dwnld_worker = RawDataDownloader()
    return dwnld_worker


# =============================================================================
# ---- Tests for the RawDataDownloader
# =============================================================================
def test_download_raw_data(dwnld_worker, workdir, qtbot):
    """
    Test that downloading raw datafiles from the Environnement Canada server
    is working as expected.
    """
    dwnld_worker = RawDataDownloader()

    # Set attributes of the data downloader.
    dwnld_worker.dirname = workdir
    dwnld_worker.StaName = "MARIEVILLE"
    dwnld_worker.stationID = "5406"
    dwnld_worker.yr_start = 2010
    dwnld_worker.yr_end = 2010
    dwnld_worker.climateID = "7024627"

    expected_filename = osp.join(
        workdir, 'RAW', "MARIEVILLE (7024627)",
        "eng-daily-01012010-12312010.csv")

    # Assert that the logic to stop the download process is working as
    # expected.
    assert not osp.exists(expected_filename)
    dwnld_worker.stop_download()
    dwnld_worker.download_data()
    assert not osp.exists(expected_filename)

    # Download data for station Marieville
    with qtbot.waitSignal(dwnld_worker.sig_download_finished):
        dwnld_worker.download_data()
    assert osp.exists(expected_filename)

    # Download the raw data file again to test the case when the raw data
    # file already exists in the 'RAW' folder.
    dwnld_worker.download_data()


def test_read_raw_datafile(workdir):
    """
    Read the weather data from raw data file that was just downloaded in the
    previous test.
    """
    rawdata_filename = osp.join(
        workdir, 'RAW', "MARIEVILLE (7024627)",
        "eng-daily-01012010-12312010.csv")
    dataset = read_raw_datafile(rawdata_filename)
    assert len(dataset) == 365
    assert (dataset.columns.values.tolist() ==
            ['Year', 'Month', 'Day', 'Max Temp (°C)',
             'Min Temp (°C)', 'Mean Temp (°C)', 'Total Precip (mm)'])
    assert (dataset.iloc[0].values.tolist() ==
            ['2010', '01', '01', '-3.0', '-6.0', '-4.5', '3.0'])


if __name__ == "__main__":
    pytest.main(['-x', __file__, '-v', '-rw'])
