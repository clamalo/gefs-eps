import xarray as xr
import asyncio
import time
import os

os.system('ulimit -n 4096')

import src.ingest.ingest as ingest
import src.processing.dataset as dataset_processing
import src.processing.points as points_processing
import src.plotting.plot_points as plot_points


date = '20241001'
cycle = '12'
starting_step = 3
ending_step = 145
delta_t = 3
eps = False
gefs = True


if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('output'):
    os.makedirs('output')


points_list = []
point_data_dict = {}
point_data_dict['metadata'] = {'date': date, 'cycle': cycle, 'starting_step': starting_step, 'ending_step': ending_step}
with open('points.txt') as f:
    for line in f:
        name, lat, lon, elevation = line.split(',')
        points_list.append([name,float(lat),float(lon),float(elevation)])
        point_data_dict[name] = {'latitude': float(lat), 'longitude': float(lon), 'elevation': float(elevation), 
                                 'exceedance_probabilities': {1: [], 3: [], 6: [], 12: [], 18: [], 24: [], 36: [], 48: [], 60: [], 100: []},
                                 'snow': [], 
                                 'total_snow': [],
                                 'tp': [],
                                 'total_tp': [],
                                 'slr': [],
                                 'temp': [],
                                 'snow_level': [],
                                 'times': []}


if delta_t == 6 and starting_step == 3:
    starting_step = 6
for step in range(starting_step,ending_step+1,delta_t):

    start = time.time()
    print(step)

    asyncio.run(ingest.ingest(date,cycle,step,gefs,eps))

    if eps:
        ds = xr.load_dataset(f'data/ecmwf_{step}.grib2', engine='cfgrib', filter_by_keys={'dataType': 'pf'})
        ds = dataset_processing.crop(ds, points_list)

    if gefs:
        gefs_ds = xr.load_dataset(f'data/gefs_{step}.grib2', engine='cfgrib', filter_by_keys={'dataType': 'pf'})
        if step != 3 and step%6==0 and delta_t==3:
            prior_gefs_ds = xr.load_dataset(f'data/gefs_{step-3}.grib2', engine='cfgrib', filter_by_keys={'dataType': 'pf'})
            gefs_ds = gefs_ds-prior_gefs_ds
        gefs_ds = gefs_ds.assign_coords(longitude=(((gefs_ds.longitude + 180) % 360) - 180)).sortby('longitude')
        gefs_ds = dataset_processing.crop(gefs_ds,points_list)

        p_gefs_ds = xr.load_dataset(f'data/p_gefs_{step}.grib2', engine='cfgrib', filter_by_keys={'dataType': 'pf'})
        p_gefs_ds = p_gefs_ds.assign_coords(longitude=(((p_gefs_ds.longitude + 180) % 360) - 180)).sortby('longitude')
        p_gefs_ds = dataset_processing.crop(p_gefs_ds,points_list)
        p_gefs_ds = p_gefs_ds.interp(latitude=gefs_ds.latitude, longitude=gefs_ds.longitude)

        gefs_ds = xr.merge([gefs_ds,p_gefs_ds])
        
        if eps:
            ds = xr.concat([ds,gefs_ds],dim='number')
        else:
            ds = gefs_ds

    point_data_dict = points_processing.process_frame(point_data_dict, ds, step, starting_step, delta_t, gefs, eps)
    print(f'Frame runtime: {time.time()-start}')

    if step%6 == 0 or step == ending_step-1:
        points_processing.interpolate_hourly(delta_t)
        plot_points.plot_points(gefs, eps)