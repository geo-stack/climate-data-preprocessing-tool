# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

"""
Tests for the mainwindow.
"""

# ---- Standard imports
import os
import os.path as osp
os.environ['CDPREP_PYTEST'] = 'True'

# ---- Third party imports
from appconfigs.base import get_home_dir
import pytest
from qtpy.QtCore import QSize

# ---- Local imports
from cdprep.config.gui import INIT_MAINWINDOW_SIZE
from cdprep.app.mainwindow import MainWindow, CONF


# =============================================================================
# ---- Fixtures
# =============================================================================
@pytest.fixture()
def conf():
    CONF.reset_to_defaults()


@pytest.fixture(scope='module')
def workdir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("test_project")
    return osp.join(tmp_path, 'mainwindow_test_project.gwp')


@pytest.fixture()
def mainwindow(qtbot, mocker, conf):
    """A fixture for Gwire mainwindow."""
    mainwindow = MainWindow()
    qtbot.addWidget(mainwindow)
    mainwindow.show()
    qtbot.waitForWindowShown(mainwindow)
    return mainwindow


# =============================================================================
# ---- Tests for MainWindow
# =============================================================================
def test_mainwindow_init(mainwindow):
    """Test that the mainwindow is initialized correctly."""
    assert mainwindow
    assert mainwindow.size() == QSize(*INIT_MAINWINDOW_SIZE)
    assert mainwindow._workdir == get_home_dir()


if __name__ == "__main__":
    pytest.main(['-x', __file__, '-v', '-rw', '-s'])
