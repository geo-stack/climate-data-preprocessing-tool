# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------

"""
File for running tests programmatically.
"""

import os
os.environ['CDPREP_PYTEST'] = 'True'

import pytest


def main():
    """
    Run pytest tests.
    """
    errno = pytest.main(['-x', 'cdprep', '-v', '-rw', '--durations=10',
                         '--cov=cdprep', '-o', 'junit_family=xunit2',
                         '--no-coverage-upload'])
    if errno != 0:
        raise SystemExit(errno)


if __name__ == '__main__':
    main()
