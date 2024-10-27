# run_model.py
import argparse
import xarray as xr
import asyncio
import time
import os
import math
import json
import signal
import sys
from datetime import timedelta, datetime  # Added datetime for time calculations
import src.ingest.ingest as ingest
import src.processing.dataset as dataset_processing
import src.processing.points as points_processing
import src.plotting.plot_points as plot_points

PROGRESS_FILE = os.path.join(os.getcwd(), 'data', 'progress.json')
terminate = False

def signal_handler(signum, frame):
    global terminate
    terminate = True
    print("Termination signal received. Exiting gracefully...")
    sys.exit(0)

# Register the signal handler for graceful termination
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def update_progress(current_step_num, total_steps, delta_t, step_durations, start_time):
    if terminate:
        return
    current_hours = current_step_num * delta_t
    total_hours = total_steps * delta_t

    # Get current time
    current_time = datetime.now()  # Get the current time

    # Calculate average duration per hour based on all steps except the first one
    if current_step_num == 1 and len(step_durations) >= 1:
        average_time_per_hour = step_durations[0]
    elif current_step_num > 1 and len(step_durations) > 1:
        average_time_per_hour = sum(step_durations[1:]) / (current_step_num - 1)
    else:
        average_time_per_hour = 0

    # Calculate remaining hours
    remaining_hours = total_hours - current_hours
    # Estimate remaining time
    estimated_remaining_seconds = average_time_per_hour * (remaining_hours/delta_t)

    # Calculate predicted finish time
    predicted_finish_time = current_time + timedelta(seconds=int(estimated_remaining_seconds))

    # Convert predicted_finish_time to an ISO formatted string
    predicted_finish_time_str = predicted_finish_time.isoformat()

    # Format as HH:MM:SS (keeping for compatibility)
    estimated_time_remaining = str(timedelta(seconds=int(estimated_remaining_seconds)))

    percentage = math.floor((current_hours / total_hours) * 100) if total_steps > 0 else 100
    percentage = max(0, min(percentage, 100))  # Ensure progress is between 0 and 100

    progress_data = {
        "percentage": percentage,
        "current": current_hours,
        "total": total_hours,
        "estimated_time_remaining": estimated_time_remaining,
        "predicted_finish_time": predicted_finish_time_str  # Added predicted_finish_time to progress data
    }
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        print(f"Error updating progress: {e}")

def main(args):
    global terminate
    date = args.date
    cycle = args.cycle
    starting_step = args.starting_step
    ending_step = args.ending_step
    delta_t = args.delta_t
    eps = args.eps
    gefs = args.gefs

    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('output'):
        os.makedirs('output')

    points_list = []
    point_data_dict = {}
    point_data_dict['metadata'] = {
        'date': date,
        'cycle': cycle,
        'starting_step': starting_step,
        'ending_step': ending_step
    }
    with open('points.txt') as f:
        for line in f:
            if line.strip() == "":
                continue  # Skip empty lines
            try:
                name, lat, lon, elevation = line.strip().split(',')
                points_list.append([name, float(lat), float(lon), float(elevation)])
                point_data_dict[name] = {
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'elevation': float(elevation),
                    'exceedance_probabilities': {1: [], 3: [], 6: [], 12: [], 18: [], 24: [], 36: [], 48: [], 60: [], 100: []},
                    'snow': [], 
                    'total_snow': [],
                    'tp': [],
                    'total_tp': [],
                    'slr': [],
                    'temp': [],
                    'snow_level': [],
                    'times': []
                }
            except ValueError:
                print(f"Invalid line in points.txt: {line.strip()}")

    if delta_t == 6 and starting_step == 3:
        starting_step = 6

    steps = list(range(starting_step, ending_step +1, delta_t))
    total_steps = len(steps)

    start_time = time.time()
    step_durations = []  # List to store duration of each step

    for step in steps:
        if terminate:
            print("Termination flag set. Exiting the model run.")
            break

        step_start_time = time.time()
        print(f"Processing step: {step}")

        try:
            asyncio.run(ingest.ingest(date, cycle, step, gefs, eps))
        except Exception as e:
            print(f"Error during ingestion for step {step}: {e}")
            step_durations.append(time.time() - step_start_time)
            continue

        if eps:
            try:
                ds = xr.load_dataset(os.path.join(os.getcwd(), 'data', f'ecmwf_{step}.grib2'), engine='cfgrib', filter_by_keys={'dataType': 'pf'})
                ds = dataset_processing.crop(ds, points_list)
            except Exception as e:
                print(f"Error loading EPS dataset for step {step}: {e}")
                step_durations.append(time.time() - step_start_time)
                continue

        if gefs:
            try:
                gefs_ds = xr.load_dataset(os.path.join(os.getcwd(), 'data', f'gefs_{step}.grib2'), engine='cfgrib', filter_by_keys={'dataType': 'pf'})
                if step != 3 and step % 6 == 0 and delta_t == 3:
                    prior_gefs_ds = xr.load_dataset(os.path.join(os.getcwd(), 'data', f'gefs_{step-3}.grib2'), engine='cfgrib', filter_by_keys={'dataType': 'pf'})
                    gefs_ds = gefs_ds - prior_gefs_ds
                gefs_ds = gefs_ds.assign_coords(longitude=(((gefs_ds.longitude + 180) % 360) - 180)).sortby('longitude')
                gefs_ds = dataset_processing.crop(gefs_ds, points_list)

                p_gefs_ds = xr.load_dataset(os.path.join(os.getcwd(), 'data', f'p_gefs_{step}.grib2'), engine='cfgrib', filter_by_keys={'dataType': 'pf'})
                p_gefs_ds = p_gefs_ds.assign_coords(longitude=(((p_gefs_ds.longitude + 180) % 360) - 180)).sortby('longitude')
                p_gefs_ds = dataset_processing.crop(p_gefs_ds, points_list)
                p_gefs_ds = p_gefs_ds.interp(latitude=gefs_ds.latitude, longitude=gefs_ds.longitude)

                gefs_ds = xr.merge([gefs_ds, p_gefs_ds])
                
                if eps:
                    ds = xr.concat([ds, gefs_ds], dim='number')
                else:
                    ds = gefs_ds
            except Exception as e:
                print(f"Error loading GEFS dataset for step {step}: {e}")
                step_durations.append(time.time() - step_start_time)
                continue

        try:
            point_data_dict = points_processing.process_frame(point_data_dict, ds, step, starting_step, delta_t, gefs, eps)
        except Exception as e:
            print(f"Error processing frame for step {step}: {e}")
            step_durations.append(time.time() - step_start_time)
            continue

        print(f'Frame runtime: {time.time() - step_start_time:.2f} seconds')
        step_duration = time.time() - step_start_time
        step_durations.append(step_duration)

        current_step_num = steps.index(step) +1  # 1-based index
        update_progress(current_step_num, total_steps, delta_t, step_durations, start_time)

        if step % 6 == 0 or step == ending_step:
            try:
                points_processing.interpolate_hourly(delta_t)
                plot_points.plot_points(gefs, eps)
            except Exception as e:
                print(f"Error during interpolation or plotting for step {step}: {e}")
                continue

    # # After processing, generate HTML files for each point
    # for point in points_list:
    #     if terminate:
    #         print("Termination flag set. Skipping HTML generation.")
    #         break
    #     name = point[0]
    #     # Implement HTML generation logic here
    #     # For example:
    #     html_content = f"""
    #     <html>
    #         <head>
    #             <title>{name} Snowfall Forecast</title>
    #             <style>
    #                 body {{
    #                     font-family: 'Roboto', sans-serif;
    #                     padding: 20px;
    #                     background-color: #f0f2f5;
    #                 }}
    #                 h1 {{
    #                     color: #333333;
    #                     text-align: center;
    #                 }}
    #             </style>
    #         </head>
    #         <body>
    #             <h1>Data for {name}</h1>
    #             <!-- Add detailed data visualization here -->
    #         </body>
    #     </html>
    #     """
    #     try:
    #         with open(os.path.join('output', f"{name}.html"), 'w') as f:
    #             f.write(html_content)
    #     except Exception as e:
    #         print(f"Error writing HTML for point {name}: {e}")

    if not terminate:
        # Set progress to 100% upon completion
        update_progress(total_steps, total_steps, delta_t, step_durations, start_time)
    else:
        # Set progress to 0% if terminated
        update_progress(0, total_steps, delta_t, step_durations, start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Snowfall Forecast Model')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--cycle', required=True, help='Cycle number')
    parser.add_argument('--starting_step', type=int, required=True, help='Starting step')
    parser.add_argument('--ending_step', type=int, required=True, help='Ending step')
    parser.add_argument('--delta_t', type=int, required=True, help='Delta time')
    parser.add_argument('--eps', type=lambda x: (str(x).lower() == 'true'), required=True, help='EPS True/False')
    parser.add_argument('--gefs', type=lambda x: (str(x).lower() == 'true'), required=True, help='GEFS True/False')

    args = parser.parse_args()
    main(args)
