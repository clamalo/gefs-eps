import xarray as xr
import numpy as np
import json
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
import pytz
import pandas as pd
from scipy.interpolate import interp1d
import os

def median_across_radius(list):
    """
    Calculate the median value across a given radius.

    Args:
        list (list): List of numerical values.

    Returns:
        np.ndarray: Median value across the radius.
    """
    return np.mean(np.array(list), axis=1)

def find_nearest(array, value):
    """
    Find the nearest value in an array.

    Args:
        array (np.ndarray): Array of numerical values.
        value (float): Value to find the nearest to in the array.

    Returns:
        int: Index of the nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_temperature_between(gh1, gh2, parameter_pair, elevation):
    """
    Find temperature at a specific elevation given two isobaric arrays.

    Args:
        gh1 (xarray.DataArray): First isobaric array.
        gh2 (xarray.DataArray): Second isobaric array.
        parameter_pair (list): List containing parameter and geopotential height keys.
        elevation (float): Elevation at which to find the temperature.

    Returns:
        float: Temperature at the specified elevation.
    """
    parameter = parameter_pair[0]
    gh = parameter_pair[1]
    temperature = gh1[parameter] + (gh2[parameter] - gh1[parameter]) * ((elevation - gh1[gh]) / (gh2[gh] - gh1[gh]))
    return temperature

def slr(point_t2ms):
    """
    Convert temperature to Fahrenheit and calculate Snow Liquid Ratio (SLR).

    Args:
        point_t2ms (xarray.DataArray): Temperature values in Kelvin.

    Returns:
        xarray.DataArray: Snow Liquid Ratio values.
    """
    
    point_slrs = xr.where(point_t2ms > 35.0, 0,
                xr.where(point_t2ms >= 28.4,
                    8 + (point_t2ms - 32.0) * (10 - 8) / (28.4 - 32.0),
                xr.where(point_t2ms >= 24.8,
                    10 + (point_t2ms - 28.4) * (11 - 10) / (24.8 - 28.4),
                xr.where(point_t2ms >= 21.2,
                    11 + (point_t2ms - 24.8) * (12 - 11) / (21.2 - 24.8),
                xr.where(point_t2ms >= 17.6,
                    12 + (point_t2ms - 21.2) * (13 - 12) / (17.6 - 21.2),
                xr.where(point_t2ms >= 14.0,
                    13 + (point_t2ms - 17.6) * (15 - 13) / (14.0 - 17.6),
                xr.where(point_t2ms >= 10.4,
                    15 + (point_t2ms - 14.0) * (20 - 15) / (10.4 - 14.0),
                xr.where(point_t2ms >= 6.8,
                    20 + (point_t2ms - 10.4) * (25 - 20) / (6.8 - 10.4),
                xr.where(point_t2ms >= 3.2,
                    25 + (point_t2ms - 6.8) * (28 - 25) / (3.2 - 6.8),
                xr.where(point_t2ms >= -0.4,
                    28 + (point_t2ms - 3.2) * (25 - 28) / (-0.4 - 3.2),
                xr.where(point_t2ms >= -4.0,
                    25 + (point_t2ms - (-0.4)) * (20 - 25) / (-4.0 - (-0.4)),
                xr.where(point_t2ms >= -7.6,
                    20 + (point_t2ms - (-4.0)) * (15 - 20) / (-7.6 - (-4.0)),
                xr.where(point_t2ms >= -11.2,
                    15 + (point_t2ms - (-7.6)) * (12 - 15) / (-11.2 - (-7.6)),
                xr.where(point_t2ms >= -14.8,
                    12 + (point_t2ms - (-11.2)) * (10 - 12) / (-14.8 - (-11.2)),
                    10
                ))))))))))))))
    return point_slrs

def find_zero_c_elevation(ghs, point_t2ms, point_elevation):
    """
    Finds the elevation of 0°C (273.15K) for each ensemble member.

    Args:
        ghs (list of xarray.DataArray): List of 5 xarray datasets containing 'gh' and 't' variables.
        point_t2ms (xarray.DataArray): Surface temperature for each ensemble member (shape: (80,)).
        point_elevation (float): Elevation of the point in meters.

    Returns:
        xarray.DataArray: Elevation of 0°C for each ensemble member (shape: (50,)).
    """
    zero_c_elevation = []

    for ensemble_idx in range(point_t2ms.shape[0]):
        isobaric_levels = [gh.isel(number=ensemble_idx) for gh in ghs]
        closest_level = min(isobaric_levels, key=lambda level: abs(level.gh.data - point_elevation))
        level_elevation = closest_level.gh.data
        level_temp = closest_level.t.data

        surface_temp = point_t2ms.data[ensemble_idx]
        lapse_rate = (level_temp - surface_temp) / (level_elevation - point_elevation)

        zero_c_elev = point_elevation + (274.26 - surface_temp) / lapse_rate 
        zero_c_elevation.append(zero_c_elev)

    return xr.DataArray(zero_c_elevation, coords={'number': range(point_t2ms.shape[0])})

def process_frame(points_data_dict, ds, step, starting_step, delta_t, gefs, eps):
    """
    Process a single frame of data and update the points data dictionary.

    Args:
        points_data_dict (dict): Dictionary containing points data.
        ds (xarray.Dataset): Dataset to process.
        step (int): Current time step.
        starting_step (int): Initial time step.
        gefs (bool): Flag indicating whether GEFS data is included.
        eps (bool): Flag indicating whether EPS data is included.

    Returns:
        dict: Updated points data dictionary.
    """

    for point in points_data_dict:
        if point == 'metadata':
            continue

        point_name, point_lat, point_lon, point_elevation, date_cycle, starting_step = (
            point, 
            points_data_dict[point]['latitude'], 
            points_data_dict[point]['longitude'], 
            points_data_dict[point]['elevation'] * 0.3048, 
            points_data_dict['metadata']['date'] + points_data_dict['metadata']['cycle'], 
            points_data_dict['metadata']['starting_step']
        )

        point_ds = ds.sel(latitude=point_lat, longitude=point_lon, method='nearest')

        # Temperature processing
        ghs = [point_ds.isel(isobaricInhPa=i) for i in range(5)]
        point_t2ms = xr.where(point_elevation < ghs[1]['gh'], 
                              find_temperature_between(ghs[0], ghs[1], ['t', 'gh'], point_elevation),
                              xr.where(point_elevation < ghs[2]['gh'], 
                                       find_temperature_between(ghs[1], ghs[2], ['t', 'gh'], point_elevation),
                                       xr.where(point_elevation < ghs[3]['gh'], 
                                                find_temperature_between(ghs[2], ghs[3], ['t', 'gh'], point_elevation),
                                                find_temperature_between(ghs[3], ghs[4], ['t', 'gh'], point_elevation))))
        
        snow_level = find_zero_c_elevation(ghs, point_t2ms, point_elevation) * 3.28084
        points_data_dict[point]['snow_level'].append(snow_level.values.tolist())

        point_t2ms = point_t2ms * 9/5 - 459.67
        points_data_dict[point]['temp'].append(point_t2ms.values.tolist())

        # Snow Liquid Ratio (SLR)
        point_slrs = slr(point_t2ms)
        points_data_dict[point]['slr'].append(point_slrs.values.tolist())
        
        # Calculate SLR exceedance probabilities
        for threshold in points_data_dict[point]['slr_exceedance_probabilities'].keys():
            percentage_above_threshold = (float((point_slrs >= threshold).sum().values) / point_slrs.size) * 100
            points_data_dict[point]['slr_exceedance_probabilities'][threshold].append(percentage_above_threshold)

        # Total Precipitation (TP)
        index_of_closest_lat = find_nearest(ds['latitude'].values, point_lat)
        index_of_closest_lon = find_nearest(ds['longitude'].values, point_lon)
        near_ds = ds.isel(latitude=slice(index_of_closest_lat-1, index_of_closest_lat+2), 
                          longitude=slice(index_of_closest_lon-1, index_of_closest_lon+2))
        near_array = near_ds['tp'].values

        tps = []
        for i in range(0, 360, 20):
            x = point_lon + 0.12 * np.cos(np.radians(i))
            y = point_lat + 0.12 * np.sin(np.radians(i))
            idx_lat = find_nearest(near_ds['latitude'].values, y)
            idx_lon = find_nearest(near_ds['longitude'].values, x)
            tp = near_array[:, idx_lat, idx_lon]
            tps.append(tp)
        tps = np.array(tps)
        point_tps = tps.mean(axis=0)
        if eps:
            point_tps[0:50] = point_tps[0:50] * 39.3701 - (0 if step == starting_step else points_data_dict[point]['total_tp'][-1][0:50])
        if gefs:
            if eps:
                point_tps[50:] = point_tps[50:] * 0.0393701
            else:
                point_tps = point_tps * 0.0393701
        point_tps = np.maximum(point_tps, 0)
        points_data_dict[point]['tp'].append(point_tps.tolist())
        point_total_tps = point_tps if step == starting_step else (np.array(points_data_dict[point]['total_tp'][-1]) + point_tps)
        points_data_dict[point]['total_tp'].append(point_total_tps.tolist())

        # Snow Calculation
        point_snows = point_tps * point_slrs
        points_data_dict[point]['snow'].append(point_snows.values.tolist())
        point_total_snows = point_snows.values.tolist() if step == starting_step else (np.array(points_data_dict[point]['total_snow'][-1]) + point_snows.values).tolist()
        points_data_dict[point]['total_snow'].append(point_total_snows)

        # Total Snow Exceedance Probability Calculation
        for threshold in points_data_dict[point]['total_snow_exceedance_probabilities'].keys():
            percentage_above_threshold = (float(np.sum(np.array(point_total_snows) >= threshold)) / len(point_total_snows)) * 100
            points_data_dict[point]['total_snow_exceedance_probabilities'][threshold].append(percentage_above_threshold)

        # Times
        lat, lon = points_data_dict[point]['latitude'], points_data_dict[point]['longitude']
        timezone = pytz.timezone(TimezoneFinder().timezone_at(lat=lat, lng=lon))
        utc_datetime_date_cycle = datetime.strptime(date_cycle, '%Y%m%d%H')
        utc_datetime_date_cycle = utc_datetime_date_cycle + timedelta(hours=starting_step-delta_t)
        datetime_date_cycle = utc_datetime_date_cycle.replace(tzinfo=pytz.utc).astimezone(timezone)
        points_data_dict[point]['times'].append((datetime_date_cycle + timedelta(hours=delta_t * (len(points_data_dict[point]['times']) + 1))).strftime('%Y-%m-%d %H:%M:%S'))

    with open(os.path.join(os.getcwd(), 'data', 'points_data.json'), 'w') as f:
        json.dump(points_data_dict, f)

    return points_data_dict

def interpolate_hourly(delta_t):
    """
    Interpolate data to an hourly frequency and save to a new JSON file.

    Loads data from 'data/points_data.json', performs interpolation on times and numerical data, and saves the modified data
    to 'data/points_data_hourly.json'.
    """
    with open(os.path.join(os.getcwd(), 'data', 'points_data.json'), 'r') as file:
        points_data_dict = json.load(file)

    def interpolate_times(times):
        time_range = pd.date_range(start=times[0], end=times[-1], freq='h')
        return [time.strftime('%Y-%m-%d %H:%M:%S') for time in time_range]

    def interpolate_data(data):
        original_steps = np.arange(0, len(data))
        new_steps = np.linspace(0, len(data) - 1, (len(data) - 1) * delta_t + 1)
        interpolated_data = [interp1d(original_steps, member_data, kind='linear')(new_steps) for member_data in data.T]
        return np.array(interpolated_data).T

    for resort in points_data_dict.keys():
        if resort == 'metadata': continue
        points_data_dict[resort]['times'] = interpolate_times(points_data_dict[resort]['times'])
        all_variables = list(points_data_dict[resort].keys())
        variables_to_exclude = ['latitude', 'longitude', 'elevation', 'times', 'total_snow_exceedance_probabilities', 'slr_exceedance_probabilities']
        variables_to_interpolate = [var for var in all_variables if var not in variables_to_exclude]
        for variable in variables_to_interpolate:
            data = np.array(points_data_dict[resort][variable])
            interpolated_data = interpolate_data(data)
            points_data_dict[resort][variable] = interpolated_data.tolist()
        
        for threshold in points_data_dict[resort]['total_snow_exceedance_probabilities'].keys():
            data = np.array(points_data_dict[resort]['total_snow_exceedance_probabilities'][threshold])
            data = data.reshape((len(data), 1))
            interpolated_data = interpolate_data(data)
            points_data_dict[resort]['total_snow_exceedance_probabilities'][threshold] = interpolated_data.tolist()
            
        for threshold in points_data_dict[resort]['slr_exceedance_probabilities'].keys():
            data = np.array(points_data_dict[resort]['slr_exceedance_probabilities'][threshold])
            data = data.reshape((len(data), 1))
            interpolated_data = interpolate_data(data)
            points_data_dict[resort]['slr_exceedance_probabilities'][threshold] = interpolated_data.tolist()

    with open(os.path.join(os.getcwd(), 'data', 'points_data_hourly.json'), 'w') as outfile:
        json.dump(points_data_dict, outfile)