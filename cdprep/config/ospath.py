# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

# ---- Standard imports
import os.path as osp

# ---- Third party imports
from appconfigs.base import get_home_dir

# ---- Local imports
from cdprep.config.main import CONF


def get_select_file_dialog_dir():
    """"
    Return the directory that should be displayed by default
    in file dialogs.
    """
    directory = CONF.get('main', 'select_file_dialog_dir', get_home_dir())
    directory = directory if osp.exists(directory) else get_home_dir()
    return directory


def set_select_file_dialog_dir(directory):
    """"
    Save in the user configs the directory that should be displayed
    by default in file dialogs.
    """
    if directory is None or not osp.exists(directory):
        directory = get_home_dir()
    CONF.set('main', 'select_file_dialog_dir', directory)
