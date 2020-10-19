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
from shutil import rmtree


def delete_folder_recursively(dirpath, delete_root=True):
    """Try to delete all files and sub-folders below the given dirpath."""
    for filename in os.listdir(dirpath):
        filepath = osp.join(dirpath, filename)
        try:
            rmtree(filepath)
        except OSError:
            os.remove(filepath)
    if delete_root:
        rmtree(dirpath)
