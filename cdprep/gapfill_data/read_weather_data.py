# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © Climate Data Preprocessing Tool Project Contributors
# https://github.com/cgq-qgc/climate-data-preprocessing-tool
#
# This file is part of Climate Data Preprocessing Tool.
# Licensed under the terms of the GNU General Public License.
# -----------------------------------------------------------------------------


# ---- Standard imports
import csv
import datetime as dt
import os.path as osp
import re
from collections import OrderedDict

# ---- Third party imports
import numpy as np
import pandas as pd


PRECIP_VARIABLES = ['Ptot', 'Rain', 'Snow']
TEMP_VARIABLES = ['Tmax', 'Tavg', 'Tmin', 'PET']
METEO_VARIABLES = PRECIP_VARIABLES + TEMP_VARIABLES
VARLABELS_MAP = {'Ptot': 'Ptot (mm)',
                 'Rain': 'Rain (mm)',
                 'Snow': 'Snow (mm)',
                 'Tmax': 'Tmax (\u00B0C)',
                 'Tavg': 'Tavg (\u00B0C)',
                 'Tmin': 'Tmin (\u00B0C)'}


def read_weather_datafile(filename):
    """
    Read the weather data from the provided filename.

    Parameters
    ----------
    filename : str
        The absolute path of an input weather data file.
    """
    metadata = {'filename': filename,
                'Station Name': '',
                'Station ID': '',
                'Location': '',
                'Latitude': 0,
                'Longitude': 0,
                'Elevation': 0}

    # Read the file.
    root, ext = osp.splitext(filename)
    if ext in ['.csv', '.out']:
        with open(filename, 'r') as csvfile:
            data = list(csv.reader(csvfile, delimiter=','))
    elif ext in ['.xls', '.xlsx']:
        data = pd.read_excel(filename, dtype='str', header=None)
        data = data.values.tolist()
    else:
        raise ValueError("Supported file format are: ",
                         ['.csv', '.out', '.xls', '.xlsx'])

    # Read the metadata and try to find the row where the
    # numerical data begin.
    header_regex_type = {
        'Station Name': (r'(stationname|name)', str),
        'Station ID': (r'(stationid|id|climateidentifier)', str),
        'Latitude': (r'(latitude)', float),
        'Longitude': (r'(longitude)', float),
        'Location': (r'(location|province)', str),
        'Elevation': (r'(elevation|altitude)', float)}
    for i, row in enumerate(data):
        if len(row) == 0 or pd.isnull(row[0]):
            continue

        label = row[0].replace(" ", "").replace("_", "")
        for key, (regex, dtype) in header_regex_type.items():
            if re.search(regex, label, re.IGNORECASE):
                try:
                    metadata[key] = dtype(row[1])
                except ValueError:
                    print("Wrong format for entry '{}'.".format(key))
                else:
                    break
        else:
            if re.search(r'(year)', label, re.IGNORECASE):
                break
    else:
        raise ValueError("Cannot find the beginning of the data.")

    # Extract and format the numerical data from the file.
    data = pd.DataFrame(data[i + 1:], columns=data[i])
    data = data.replace(r'(?i)^\s*$|nan|none', np.nan, regex=True)

    # The data must contain the following columns :
    # (1) Tmax, (2) Tavg, (3) Tmin, (4) Ptot.
    # The dataframe can also have these optional columns:
    # (5) Rain, (6) Snow, (7) PET
    # The dataframe must use a datetime index.

    column_names_regexes = OrderedDict([
        ('Year', r'(year)'),
        ('Month', r'(month)'),
        ('Day', r'(day)'),
        ('Tmax', r'(maxtemp)'),
        ('Tmin', r'(mintemp)'),
        ('Tavg', r'(meantemp)'),
        ('Ptot', r'(totalprecip)'),
        ('Rain', r'(rain)'),
        ('Snow', r'(snow)')])
    for i, column in enumerate(data.columns):
        column_ = column.replace(" ", "").replace("_", "")
        for key, regex in column_names_regexes.items():
            if re.search(regex, column_, re.IGNORECASE):
                data = data.rename(columns={column: key})
                break
        else:
            data = data.drop([column], axis=1)

    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            print("Some {} data could not be converted to numeric value"
                  .format(col))

    # We now create the time indexes for the dataframe form the year,
    # month, and day data.
    data = data.set_index(pd.to_datetime(dict(
        year=data['Year'], month=data['Month'], day=data['Day'])))
    data = data.drop(labels=['Year', 'Month', 'Day'], axis=1)
    data.index.names = ['Datetime']

    # We print some comment if optional data was loaded from the file.
    if 'Rain' in data.columns:
        print('Rain data imported from datafile.')
    if 'Snow' in data.columns:
        print('Snow data imported from datafile.')

    return metadata, data


def open_weather_log(fname):
    """
    Open the csv file and try to guess the delimiter.
    Return None if this fails.
    """
    for dlm in [',', '\t']:
        with open(fname, 'r') as f:
            reader = list(csv.reader(f, delimiter=dlm))
            if reader[0][0] == 'Station Name':
                return reader[36:]
    else:
        return None


def load_weather_log(fname, varname):
    reader = open_weather_log(fname)
    datetimes = []
    for i in range(len(reader)):
        if reader[i][0] == varname:
            year = int(float(reader[i][1]))
            month = int(float(reader[i][2]))
            day = int(float(reader[i][3]))
            datetimes.append(dt.datetime(year, month, day))
    return pd.DatetimeIndex(datetimes)


def calcul_rain_from_ptot(Tavg, Ptot, Tcrit=0):
    rain = Ptot.copy(deep=True)
    rain[Tavg < Tcrit] = 0

    # np.copy(Ptot)
    # rain[np.where(Tavg < Tcrit)[0]] = 0
    return rain


def generate_weather_HTML(staname, prov, lat, climID, lon, alt):

    # HTML table with the info related to the weather station.

    FIELDS = [['Station', staname],
              ['Latitude', '%0.3f°' % lat],
              ['Longitude', '%0.3f°' % lon],
              ['Altitude', '%0.1f m' % alt],
              ['Clim. ID', climID],
              ['Province', prov]
              ]

    table = '<table border="0" cellpadding="2" cellspacing="0" align="left">'
    for row in FIELDS:
        table += '''
                 <tr>
                   <td width=10></td>
                   <td align="left">%s</td>
                   <td align="left" width=20>:</td>
                   <td align="left">%s</td>
                   </tr>
                 ''' % (row[0], row[1])
    table += '</table>'

    return table
