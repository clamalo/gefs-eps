import xarray as xr
import numpy as np
import json
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
import pytz
import pandas as pd
from scipy.interpolate import interp1d
import os

_RADIUS_ANGLES = np.radians(np.arange(0, 360, 20))
_RADIUS_OFFSET = 0.12


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
    temperature = gh1[parameter] + (gh2[parameter] - gh1[parameter]) * (
        (elevation - gh1[gh]) / (gh2[gh] - gh1[gh])
    )
    return temperature


def slr(point_t2ms):
    """
    Convert temperature to Fahrenheit and calculate Snow Liquid Ratio (SLR).

    Args:
        point_t2ms (xarray.DataArray): Temperature values in Kelvin.

    Returns:
        np.ndarray: Snow Liquid Ratio values.
    """
    temps = np.asarray(point_t2ms)
    point_slrs = np.where(
        temps > 35.0,
        0,
        np.where(
            temps >= 28.4,
            8 + (temps - 32.0) * (10 - 8) / (28.4 - 32.0),
            np.where(
                temps >= 24.8,
                10 + (temps - 28.4) * (11 - 10) / (24.8 - 28.4),
                np.where(
                    temps >= 21.2,
                    11 + (temps - 24.8) * (12 - 11) / (21.2 - 24.8),
                    np.where(
                        temps >= 17.6,
                        12 + (temps - 21.2) * (13 - 12) / (17.6 - 21.2),
                        np.where(
                            temps >= 14.0,
                            13 + (temps - 17.6) * (15 - 13) / (14.0 - 17.6),
                            np.where(
                                temps >= 10.4,
                                15 + (temps - 14.0) * (20 - 15) / (10.4 - 14.0),
                                np.where(
                                    temps >= 6.8,
                                    20 + (temps - 10.4) * (25 - 20) / (6.8 - 10.4),
                                    np.where(
                                        temps >= 3.2,
                                        25 + (temps - 6.8) * (28 - 25) / (3.2 - 6.8),
                                        np.where(
                                            temps >= -0.4,
                                            28
                                            + (temps - 3.2) * (25 - 28) / (-0.4 - 3.2),
                                            np.where(
                                                temps >= -4.0,
                                                25
                                                + (temps - (-0.4))
                                                * (20 - 25)
                                                / (-4.0 - (-0.4)),
                                                np.where(
                                                    temps >= -7.6,
                                                    20
                                                    + (temps - (-4.0))
                                                    * (15 - 20)
                                                    / (-7.6 - (-4.0)),
                                                    np.where(
                                                        temps >= -11.2,
                                                        15
                                                        + (temps - (-7.6))
                                                        * (12 - 15)
                                                        / (-11.2 - (-7.6)),
                                                        np.where(
                                                            temps >= -14.8,
                                                            12
                                                            + (temps - (-11.2))
                                                            * (10 - 12)
                                                            / (-14.8 - (-11.2)),
                                                            10,
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return point_slrs


def _interpolate_temperature_at_elevation(gh_levels, t_levels, elevation):
    temp_01 = t_levels[0] + (t_levels[1] - t_levels[0]) * (
        (elevation - gh_levels[0]) / (gh_levels[1] - gh_levels[0])
    )
    temp_12 = t_levels[1] + (t_levels[2] - t_levels[1]) * (
        (elevation - gh_levels[1]) / (gh_levels[2] - gh_levels[1])
    )
    temp_23 = t_levels[2] + (t_levels[3] - t_levels[2]) * (
        (elevation - gh_levels[2]) / (gh_levels[3] - gh_levels[2])
    )
    temp_34 = t_levels[3] + (t_levels[4] - t_levels[3]) * (
        (elevation - gh_levels[3]) / (gh_levels[4] - gh_levels[3])
    )

    return np.where(
        elevation < gh_levels[1],
        temp_01,
        np.where(
            elevation < gh_levels[2],
            temp_12,
            np.where(elevation < gh_levels[3], temp_23, temp_34),
        ),
    )


def find_zero_c_elevation(gh_levels, t_levels, point_t2ms, point_elevation):
    """
    Finds the elevation of 0°C (273.15K) for each ensemble member.

    Args:
        gh_levels (np.ndarray): Geopotential heights (shape: (5, members)).
        t_levels (np.ndarray): Temperatures (shape: (5, members)).
        point_t2ms (np.ndarray): Surface temperature for each ensemble member (shape: (members,)).
        point_elevation (float): Elevation of the point in meters.

    Returns:
        np.ndarray: Elevation of 0°C for each ensemble member (shape: (members,)).
    """
    diffs = np.abs(gh_levels - point_elevation)
    closest_idx = diffs.argmin(axis=0)
    member_idx = np.arange(point_t2ms.shape[0])
    level_elevation = gh_levels[closest_idx, member_idx]
    level_temp = t_levels[closest_idx, member_idx]

    surface_temp = point_t2ms
    lapse_rate = (level_temp - surface_temp) / (level_elevation - point_elevation)

    zero_c_elev = point_elevation + (274.26 - surface_temp) / lapse_rate
    return zero_c_elev


def build_ensemble_slices(total_members, metadata, eps, aifs, gefs):
    ensemble_order = metadata.get("ensemble_order", [])
    ensemble_counts = metadata.get("ensemble_counts", {})
    slices = []

    if isinstance(ensemble_order, list) and isinstance(ensemble_counts, dict):
        offset = 0
        for name in ensemble_order:
            count = int(ensemble_counts.get(name, 0) or 0)
            if count <= 0:
                continue
            end = min(offset + count, total_members)
            if end > offset:
                slices.append((name, slice(offset, end)))
            offset = end

        if slices:
            return slices

    offset = 0
    if eps:
        end = min(offset + 50, total_members)
        if end > offset:
            slices.append(("eps", slice(offset, end)))
        offset = end
    if gefs:
        end = min(offset + 30, total_members)
        if end > offset:
            slices.append(("gefs", slice(offset, end)))
    if aifs and not slices and total_members > 0:
        slices.append(("aifs", slice(0, total_members)))

    return slices


def process_frame(points_data_dict, ds, step, starting_step, delta_t, gefs, eps, aifs):
    """
    Process a single frame of data and update the points data dictionary.

    Args:
        points_data_dict (dict): Dictionary containing points data.
        ds (xarray.Dataset): Dataset to process.
        step (int): Current time step.
        starting_step (int): Initial time step.
        gefs (bool): Flag indicating whether GEFS data is included.
        eps (bool): Flag indicating whether EPS data is included.
        aifs (bool): Flag indicating whether AIFS-E data is included.

    Returns:
        dict: Updated points data dictionary.
    """

    tz_finder = TimezoneFinder()
    tz_cache = {}
    lat_values = ds["latitude"].values
    lon_values = ds["longitude"].values

    for point in points_data_dict:
        if point == "metadata":
            continue

        point_name, point_lat, point_lon, point_elevation, date_cycle, starting_step = (
            point,
            points_data_dict[point]["latitude"],
            points_data_dict[point]["longitude"],
            points_data_dict[point]["elevation"] * 0.3048,
            points_data_dict["metadata"]["date"]
            + points_data_dict["metadata"]["cycle"],
            points_data_dict["metadata"]["starting_step"],
        )

        point_lat_idx = find_nearest(lat_values, point_lat)
        point_lon_idx = find_nearest(lon_values, point_lon)
        point_ds = ds.isel(latitude=point_lat_idx, longitude=point_lon_idx)

        # Temperature processing
        gh_values = point_ds["gh"].values
        t_values = point_ds["t"].values
        gh_levels = gh_values.T
        t_levels = t_values.T

        point_t2ms = _interpolate_temperature_at_elevation(
            gh_levels, t_levels, point_elevation
        )

        snow_level = (
            find_zero_c_elevation(gh_levels, t_levels, point_t2ms, point_elevation)
            * 3.28084
        )
        points_data_dict[point]["snow_level"].append(snow_level.tolist())

        point_t2ms = point_t2ms * 9 / 5 - 459.67
        points_data_dict[point]["temp"].append(point_t2ms.tolist())

        # Snow Liquid Ratio (SLR)
        point_slrs = slr(point_t2ms)
        points_data_dict[point]["slr"].append(point_slrs.tolist())

        # Calculate SLR exceedance probabilities
        for threshold in points_data_dict[point]["slr_exceedance_probabilities"].keys():
            percentage_above_threshold = (
                float(np.sum(point_slrs >= threshold)) / point_slrs.size
            ) * 100
            points_data_dict[point]["slr_exceedance_probabilities"][threshold].append(
                percentage_above_threshold
            )

        # Total Precipitation (TP)
        near_ds = ds.isel(
            latitude=slice(point_lat_idx - 1, point_lat_idx + 2),
            longitude=slice(point_lon_idx - 1, point_lon_idx + 2),
        )
        near_array = near_ds["tp"].values
        near_lat = near_ds["latitude"].values
        near_lon = near_ds["longitude"].values

        circle_lons = point_lon + _RADIUS_OFFSET * np.cos(_RADIUS_ANGLES)
        circle_lats = point_lat + _RADIUS_OFFSET * np.sin(_RADIUS_ANGLES)
        idx_lat = np.abs(near_lat[:, None] - circle_lats[None, :]).argmin(axis=0)
        idx_lon = np.abs(near_lon[:, None] - circle_lons[None, :]).argmin(axis=0)
        point_tps = near_array[:, idx_lat, idx_lon].mean(axis=1)
        prev_total = None
        if step != starting_step and points_data_dict[point]["total_tp"]:
            prev_total = np.array(points_data_dict[point]["total_tp"][-1])

        ensemble_slices = build_ensemble_slices(
            point_tps.shape[0], points_data_dict.get("metadata", {}), eps, aifs, gefs
        )

        for name, member_slice in ensemble_slices:
            if member_slice.start >= member_slice.stop:
                continue
            if name in ("eps", "aifs"):
                converted = point_tps[member_slice] * 39.3701
                if prev_total is not None:
                    converted = converted - prev_total[member_slice]
                point_tps[member_slice] = converted
            elif name == "gefs":
                point_tps[member_slice] = point_tps[member_slice] * 0.0393701
        point_tps = np.maximum(point_tps, 0)
        points_data_dict[point]["tp"].append(point_tps.tolist())
        point_total_tps = (
            point_tps
            if step == starting_step
            else (np.array(points_data_dict[point]["total_tp"][-1]) + point_tps)
        )
        points_data_dict[point]["total_tp"].append(point_total_tps.tolist())

        # Snow Calculation
        point_snows = point_tps * point_slrs
        points_data_dict[point]["snow"].append(point_snows.tolist())
        point_total_snows = (
            point_snows.tolist()
            if step == starting_step
            else (
                np.array(points_data_dict[point]["total_snow"][-1]) + point_snows
            ).tolist()
        )
        points_data_dict[point]["total_snow"].append(point_total_snows)

        # Total Snow Exceedance Probability Calculation
        for threshold in points_data_dict[point][
            "total_snow_exceedance_probabilities"
        ].keys():
            percentage_above_threshold = (
                float(np.sum(np.array(point_total_snows) >= threshold))
                / len(point_total_snows)
            ) * 100
            points_data_dict[point]["total_snow_exceedance_probabilities"][
                threshold
            ].append(percentage_above_threshold)

        # Times
        lat, lon = (
            points_data_dict[point]["latitude"],
            points_data_dict[point]["longitude"],
        )
        tz_key = (lat, lon)
        if tz_key not in tz_cache:
            tz_name = tz_finder.timezone_at(lat=lat, lng=lon)
            tz_cache[tz_key] = pytz.timezone(tz_name) if tz_name else pytz.utc
        timezone = tz_cache[tz_key]
        utc_datetime_date_cycle = datetime.strptime(date_cycle, "%Y%m%d%H")
        utc_datetime_date_cycle = utc_datetime_date_cycle + timedelta(
            hours=starting_step - delta_t
        )
        datetime_date_cycle = utc_datetime_date_cycle.replace(
            tzinfo=pytz.utc
        ).astimezone(timezone)
        time_index = len(points_data_dict[point]["times"]) + 1
        points_data_dict[point]["times"].append(
            (datetime_date_cycle + timedelta(hours=delta_t * time_index)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    with open(os.path.join(os.getcwd(), "data", "points_data.json"), "w") as f:
        json.dump(points_data_dict, f)

    return points_data_dict


def interpolate_hourly(delta_t):
    """
    Interpolate data to an hourly frequency and save to a new JSON file.

    Loads data from 'data/points_data.json', performs interpolation on times and numerical data, and saves the modified data
    to 'data/points_data_hourly.json'.
    """
    with open(os.path.join(os.getcwd(), "data", "points_data.json"), "r") as file:
        points_data_dict = json.load(file)

    def interpolate_times(times):
        if len(times) < 2:
            return times
        time_range = pd.date_range(start=times[0], end=times[-1], freq="h")
        return [time.strftime("%Y-%m-%d %H:%M:%S") for time in time_range]

    def interpolate_data(data):
        if data.shape[0] < 2:
            return data
        original_steps = np.arange(0, len(data))
        new_steps = np.linspace(0, len(data) - 1, (len(data) - 1) * delta_t + 1)
        interpolated_data = [
            interp1d(original_steps, member_data, kind="linear")(new_steps)
            for member_data in data.T
        ]
        return np.array(interpolated_data).T

    for resort in points_data_dict.keys():
        if resort == "metadata":
            continue
        points_data_dict[resort]["times"] = interpolate_times(
            points_data_dict[resort]["times"]
        )
        all_variables = list(points_data_dict[resort].keys())
        variables_to_exclude = [
            "latitude",
            "longitude",
            "elevation",
            "times",
            "total_snow_exceedance_probabilities",
            "slr_exceedance_probabilities",
        ]
        variables_to_interpolate = [
            var for var in all_variables if var not in variables_to_exclude
        ]
        for variable in variables_to_interpolate:
            data = np.array(points_data_dict[resort][variable])
            interpolated_data = interpolate_data(data)
            points_data_dict[resort][variable] = interpolated_data.tolist()

        for threshold in points_data_dict[resort][
            "total_snow_exceedance_probabilities"
        ].keys():
            data = np.array(
                points_data_dict[resort]["total_snow_exceedance_probabilities"][
                    threshold
                ]
            )
            data = data.reshape((len(data), 1))
            interpolated_data = interpolate_data(data)
            points_data_dict[resort]["total_snow_exceedance_probabilities"][
                threshold
            ] = interpolated_data.tolist()

        for threshold in points_data_dict[resort][
            "slr_exceedance_probabilities"
        ].keys():
            data = np.array(
                points_data_dict[resort]["slr_exceedance_probabilities"][threshold]
            )
            data = data.reshape((len(data), 1))
            interpolated_data = interpolate_data(data)
            points_data_dict[resort]["slr_exceedance_probabilities"][threshold] = (
                interpolated_data.tolist()
            )

    with open(
        os.path.join(os.getcwd(), "data", "points_data_hourly.json"), "w"
    ) as outfile:
        json.dump(points_data_dict, outfile)
