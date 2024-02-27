import os
import xarray as xr
import numpy as np
from stompy.spatial import proj_utils

from shapely import prepared, geometry

def fetch_or_load_daily_avhrr_cropped():
    cache_fn="avhrr-1day-cropped.nc"
    if not os.path.exists(cache_fn):
        # Opendap for AVHRR 1-day composite
        avhrr_1d_url="https://thredds.cencoos.org/thredds/dodsC/ERDATSSTA1DAY.nc"
       
        ds=xr.open_dataset(avhrr_1d_url)
        time_sel = slice(*np.searchsorted(ds.time.values,[np.datetime64("2022-07-01"),np.datetime64("2022-09-01")]))
        lat_sel = slice(*np.searchsorted(ds.latitude.values, [37.3, 38.3]))
        lon_sel = slice(*np.searchsorted(ds.longitude.values, [360-123.3,360-121.8]))
        ds_subset = ds.isel(altitude=0,time=time_sel, latitude=lat_sel, longitude=lon_sel)
        ds_subset.to_netcdf(cache_fn)
        ds_subset.close()

    ds=xr.load_dataset(cache_fn)
    # Add utm coordinates
    _,lon,lat = xr.broadcast( ds.sst.isel(time=0), ds.longitude, ds.latitude)
    utm_xy = proj_utils.mapper('WGS84','EPSG:26910')(lon,lat)
    
    ds['x'] = lon.dims, utm_xy[...,0]
    ds['y'] = lon.dims, utm_xy[...,1]
    ds=ds.set_coords(['x','y'])

    # 2022-07-25T12:00 and 2022-07-27T12:00 are repeated? 26th omitted.
    # The repeated time stamps have the same data.
    time_sel = np.r_[True, np.diff(ds.time.values)>np.timedelta64(0,'s')]
    ds=ds.isel(time=time_sel)

    return ds

def add_watermask(ds,grid):
    water=np.zeros(ds.x.shape,bool)
    grid_poly = grid.boundary_polygon()
    poly = prepared.prep(grid_poly)

    for idx in np.ndindex(ds.x.shape):
        point=geometry.Point(ds.x[idx], ds.y[idx])
        water[idx] = poly.intersects(point)

    ds['water']=ds.x.dims,water
    # For linearizing just the water pixels:
    idxs=np.full(ds.water.shape, -1)
    idxs[water] = np.arange(water.sum())
    # so ds.water_index[row,col] => linear index.
    ds['water_index']=('latitude','longitude'),idxs
    longs,lats = np.meshgrid( np.arange(ds.dims['longitude']), np.arange(ds.dims['latitude']) )
    ds['water_ll_idx']=('wet','two'), np.array( [lats[water],longs[water]]).T

    
