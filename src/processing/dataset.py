import xarray as xr

# takes a dataset and a list of points and crops the dataset to a box around the maximum extent of the points with a 3 degree buffer on all sides
def crop(ds,points):
    max_lat = max([point[1] for point in points])+3
    min_lat = min([point[1] for point in points])-3
    max_lon = max([point[2] for point in points])+3
    min_lon = min([point[2] for point in points])-3

    ds = ds.sel(latitude=slice(max_lat,min_lat),longitude=slice(min_lon,max_lon))

    return ds

# takes a high resolution (ds1) and a low resolution (ds2) dataset and combines them at the same resolution.
# these datasets must be the same spatial coverage
def combine(ds1, ds2):
    #use interp_like to interpolate the high resolution dataset to the low resolution dataset
    ds2 = ds2.interp_like(ds1)
    return xr.merge([ds1, ds2])