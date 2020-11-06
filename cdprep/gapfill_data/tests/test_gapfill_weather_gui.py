# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

"""
Tests for the WeatherDataGapfiller.
"""

# ---- Standard imports
import os
import os.path as osp
import shutil
os.environ['CDPREP_PYTEST'] = 'True'

# ---- Third party imports
from appconfigs.base import get_home_dir
import pytest
from qtpy.QtCore import QSize

# ---- Local imports
from cdprep.gapfill_data.gapfill_weather_gui import WeatherDataGapfiller
from cdprep.app.mainwindow import MainWindow, CONF


# =============================================================================
# ---- Fixtures
# =============================================================================
@pytest.fixture()
def conf():
    CONF.reset_to_defaults()


@pytest.fixture(scope='module')
def testfiles():
    """A fixture of a list of available test files."""
    filenames = [
        "IBERVILLE (7023270)_2000-2015.csv",
        "L'ACADIE (702LED4)_2000-2015.csv",
        "MARIEVILLE (7024627)_2000-2015.csv"]
    return [
        osp.join(osp.dirname(__file__), 'data', filename) for
        filename in filenames]


@pytest.fixture(scope='module')
def workdir(tmp_path_factory, testfiles):
    """A fixture for a test working directory."""
    workdir = osp.join(
        tmp_path_factory.getbasetemp(), "@ tèst-fïl! 'dätèt!")
    os.makedirs(workdir, exist_ok=True)
    for testfile in testfiles:
        shutil.copyfile(
            testfile, osp.join(workdir, osp.basename(testfile)))
    return workdir


@pytest.fixture()
def datagapfiller(qtbot, mocker, conf):
    """A fixture for the WeatherDataGapfiller."""
    datagapfiller = WeatherDataGapfiller()
    qtbot.addWidget(datagapfiller)
    datagapfiller.show()
    qtbot.waitForWindowShown(datagapfiller)

    assert datagapfiller.Nmax.minimum() == 1

    return datagapfiller


# =============================================================================
# ---- Tests for the WeatherDataGapfiller
# =============================================================================
def test_load_data(datagapfiller, workdir, qtbot):
    """
    Test that loading input data automatically from the working directory
    is working as expected.
    """
    assert datagapfiller
    assert datagapfiller.workdir is None
    assert datagapfiller.gapfill_manager.count() == 0

    # Set the working directory to the test directory that contains
    # 3 valid datafiles.
    assert datagapfiller._loading_data_inprogress is False
    datagapfiller.set_workdir(workdir)
    assert datagapfiller._loading_data_inprogress is True
    qtbot.waitUntil(lambda: datagapfiller._loading_data_inprogress is False)
    assert datagapfiller._corrcoeff_update_inprogress is True
    qtbot.waitUntil(
        lambda: datagapfiller._corrcoeff_update_inprogress is False)

    assert datagapfiller.workdir == workdir
    assert osp.basename(workdir) == "@ tèst-fïl! 'dätèt!"
    assert datagapfiller.gapfill_manager.count() == 3

    # Set the working directory to a directory that doesn't contain any
    # valid datafile.
    datagapfiller.set_workdir(osp.dirname(workdir))
    assert datagapfiller._loading_data_inprogress is True
    qtbot.waitUntil(lambda: datagapfiller._loading_data_inprogress is False)
    assert datagapfiller._corrcoeff_update_inprogress is False

    assert datagapfiller.workdir == osp.dirname(workdir)
    assert datagapfiller.gapfill_manager.count() == 0


if __name__ == "__main__":
    pytest.main(['-x', __file__, '-v', '-rw', '-s'])
