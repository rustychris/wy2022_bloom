import pandas as pd
import os
import numpy as np

from stompy.spatial import proj_utils
from stompy import memoize
from stompy.io.local import noaa_coops, usgs_nwis

wy2022_bloom = os.path.dirname(__file__)
l2m=proj_utils.mapper('WGS84','EPSG:26910')

# Getting a unified dataframe of relevant in-situ temperature observations

time_range=[np.datetime64("2022-07-01"),np.datetime64("2022-10-01")]

def msp_to_df(station, xy):
    df = pd.read_csv(os.path.join(wy2022_bloom,f"../Kd_2022/L2/{station}_all_data_L2.csv"),
                     parse_dates={'time_pst':['dt']},low_memory=False)
    df['time'] = df.time_pst.dt.tz_convert(tz=None) # should get it into UTC
    df = df[ ['time','T_degC', 'Depth_m'] ]
    df = df[ (df.time>time_range[0]) & (df.time<time_range[1])]
    df['x'], df['y'] = xy
    df['station']=station
    return df

def noaa_to_df(station,station_id):
    noaa_ds = (noaa_coops.coops_dataset(station_id, time_range[0], time_range[1], 
                                        ['water_temperature','air_temperature','water_level'],
                                         cache_dir='cache')
                         .isel(station=0))
    noaa_df = noaa_ds.to_dataframe().reset_index()
    noaa_df['station'] = station
    noaa_df=noaa_df.rename(dict(water_temperature='T_degC', water_level='Depth_m'), axis=1)
    noaa_df['x'],noaa_df['y'] = l2m([noaa_ds.lon.values,noaa_ds.lat.values])
    return noaa_df[ ['time','station','T_degC','Depth_m','x','y'] ]


def usgs_to_df(stn,xy,station,ts_code=None):
    ds = usgs_nwis.nwis_dataset(stn,time_range[0], time_range[1],
                                [10],cache_dir='cache',
                                name_with_ts_code=(ts_code is not None))
    df=ds.to_dataframe().reset_index()
    if ts_code is None:
        rename={'temperature_water':'T_degC'}
    else:
        rename={'temperature_water_'+str(ts_code):'T_degC'}
        
    df=df.rename(rename,axis=1)
    df=df[ ['time','T_degC'] ]
    df['x'],df['y'] = xy
    df['station']=station
    return df

@memoize.memoize()
def all_stations():
    station_dfs=[]

    station_dfs.append(msp_to_df('SM', np.r_[566900, 4.16085e6]))
    station_dfs.append(msp_to_df('SHL', l2m([-122.243, 37.63079])))
    station_dfs.append(msp_to_df('SLM', l2m([-122.218, 37.6742])))
    station_dfs.append(msp_to_df('HAY', l2m([-122.201, 37.61174])))
    station_dfs.append(msp_to_df('DMB', l2m([-122.119, 37.50417])))

    station_dfs.append( noaa_to_df('Alameda', 9414750) )
    station_dfs.append( noaa_to_df('Redwood', 9414523) )
    station_dfs.append( noaa_to_df('Fort Point', 9414290) )
    station_dfs.append( noaa_to_df('Richmond', 9414863) )
    station_dfs.append( noaa_to_df('Martinez',9415102) )
    station_dfs.append( noaa_to_df('Point Reyes',9415020) )

    station_dfs.append( usgs_to_df( 374938122251801, np.r_[550030,4.18629e6], 'Alcatraz') )
    station_dfs.append( usgs_to_df( 374811122235001, np.r_[553056, 4184134], 'Pier17') )
    station_dfs.append( usgs_to_df( 11162765, np.r_[566900, 4.16085e6], 'SMB_USGS') )

    # USGS Richmond Bridge
    station_dfs.append( usgs_to_df(375607122264701,l2m(-122.4463889, 37.93527778),
                                   'Richmond Br USGS', 17273))
    
    insitu_df = pd.concat(station_dfs)
    return insitu_df

