# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © 2014-2018 GWHAT Project Contributors
# https://github.com/jnsebgosselin/gwhat
#
# This file is part of GWHAT (Ground-Water Hydrograph Analysis Toolbox).
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

"""Qt utilities"""

# ---- Third party imports
from PyQt5.QtCore import QByteArray, Qt
from PyQt5.QtWidgets import QWidget, QSizePolicy, QFrame


def create_separator(orientation):
    frame = QFrame()
    if orientation == Qt.Vertical:
        frame.setFrameStyle(53)
    else:
        frame.setFrameStyle(52)
    return frame


def qbytearray_to_hexstate(qba):
    """Convert QByteArray object to a str hexstate."""
    return str(bytes(qba.toHex().data()).decode())


def hexstate_to_qbytearray(hexstate):
    """Convert a str hexstate to a QByteArray object."""
    return QByteArray().fromHex(str(hexstate).encode('utf-8'))


def create_toolbar_stretcher():
    """Create a stretcher to be used in a toolbar """
    stretcher = QWidget()
    stretcher.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return stretcher
