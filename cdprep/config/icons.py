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
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon
import qtawesome as qta

# ---- Local imports
from cdprep import __rootdir__
from cdprep.config.gui import ICON_COLOR, RED

DIRNAME = os.path.join(__rootdir__, 'ressources', 'icons')
LOCAL_ICONS = {
    'master': 'cdprep'}

FA_ICONS = {
    'chevron_down': [
        ('mdi.chevron-down',),
        {'color': ICON_COLOR}],
    'chevron_right': [
        ('mdi.chevron-right',),
        {'color': ICON_COLOR}],
    'download_data': [
        ('mdi.download',),
        {'color': ICON_COLOR}],
    'delete_data': [
        ('mdi.delete-outline',),
        {'color': ICON_COLOR, 'scale_factor': 1.4}],
    'folder_open': [
        ('mdi.folder-open-outline',),
        {'color': ICON_COLOR, 'scale_factor': 1.3}],
    'merge_data': [
        ('mdi.table-merge-cells',),
        {'color': ICON_COLOR, 'scale_factor': 1.4}],
    'refresh': [
        ('mdi.refresh',),
        {'color': ICON_COLOR}],
    'save': [
        ('fa.save',),
        {'color': ICON_COLOR}],
    'stop': [
        ('mdi.stop-circle-outline',),
        {'color': RED}],
    }

ICON_SIZES = {'large': (32, 32),
              'normal': (28, 28),
              'small': (20, 20)}


def get_icon(name):
    """Return a QIcon from a specified icon name."""
    if name in FA_ICONS:
        args, kwargs = FA_ICONS[name]
        return qta.icon(*args, **kwargs)
    elif name in LOCAL_ICONS:
        return QIcon(osp.join(DIRNAME, LOCAL_ICONS[name]))
    else:
        return QIcon()


def get_iconsize(size):
    return QSize(*ICON_SIZES[size])
