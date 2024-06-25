# Replace met forcing - I don't have other humidity or cloudiness data, so will just
# overwrite the temperature field.
import stompy.model.delft.io as dio
from stompy.io.local import noaa_coops
from stompy import utils
from stompy.plot import plot_utils
from stompy.spatial import proj_utils, field
import xarray as xr
import numpy as np

class MetUpdater:
    noaa_air_stations=[ 9414750, # Alameda
                        9414290, # SF Fort Point
                        9414863, # Richmond
                        9414523, # Redwood City
                        9415141, # Davis Point (near Carquinez Br)
                        #9415020, # Point Reyes
                      ]
    noaa_station_names={9414750:"Alameda",
                        9414290:"SF Fort Point",
                        9414863:"Richmond",
                        9414523:"Redwood City",
                        9415141:"Davis Point (Carquinez Br)",
                        # 9415020:"Point Reyes"
                       }

    def __init__(self,original_met_fn, run_dir, new_met_fn):
        self.noaa_air_stations = list(self.noaa_air_stations) # don't modify originals
        self.noaa_station_names = dict(self.noaa_station_names)
        
        self.original_met_fn = original_met_fn
        self.run_dir = run_dir
        self.new_met_fn = new_met_fn
        self.met = dio.read_meteo_on_curvilinear(self.original_met_fn,self.run_dir,flip_grid_rows=True)

        self.load_noaa_data()

        temp_new = self.interpolate_new_temperature()

        # Update data and write back out
        self.met_new = self.met.copy()
        self.met_new['air_temperature'] = self.met_new.air_temperature.dims, temp_new
        dio.rewrite_meteo_on_curvilinear(original_met_fn,new_met_fn,self.met_new)
        
    def load_noaa_data(self):
        noaa_air_temps = [ noaa_coops.coops_dataset(station,self.met.time.values[0], self.met.time.values[-1],
                                                    ['air_temperature'],cache_dir='cache')
                           for station in self.noaa_air_stations]

        # Duplicate SF to a point to the south to keep Redwood City from bleeding out into the
        # ocean.
        def duplicate(original, new_ll):
            dupe = [ds for ds in noaa_air_temps if int(ds.station.values[0])==original][0]
            dupe = dupe.copy()
            dupe['lon'] = dupe.lon.dims, [new_ll[0]]
            dupe['lat'] = dupe.lat.dims, [new_ll[1]]
            dupe['station'] = dupe.station.dims, [-original]
            noaa_air_temps.append(dupe)
            self.noaa_air_stations.append( -original )
            self.noaa_station_names[-original] = self.noaa_station_names[original]+" dupe"
        duplicate(9414290, [-122.5,37.5068] ) # half-moon-bay ish
        duplicate(9414290, [-122.9,37.93] ) # point reyes-ish
        
        self.noaa_air_temp = xr.concat(noaa_air_temps,dim='station')
        self.noaa_air_temp['air_temperature'] = (self.noaa_air_temp.air_temperature.dims, 
                                                 utils.fill_invalid(self.noaa_air_temp.air_temperature.values,axis=1))
    def plot_original(self,tidx=0):
        fig,ax=plt.subplots()
        snap=self.met.isel(time=tidx)    
        coll=plot_utils.pad_pcolormesh( met.x, met.y, snap['air_temperature'], ax=ax,
                                       cmap=scmap.load_gradient('hot_desaturated.cpt'),
                                        clim=[13,20.5])
        #grid.plot_edges(color='k',lw=0.5,alpha=0.3,zorder=2)
        return fig,ax

    def plot_something(self):
        fig,ax=plt.subplots(figsize=(9.5,6))
        
        for stn_idx,stn in enumerate(self.noaa_air_temp.station):
            noaa_station=self.noaa_air_temp.isel(station=stn_idx)
            name=self.noaa_station_names[int(stn.item())]
            ls=ax.plot(noaa_station.time, noaa_station['air_temperature'], label=name,lw=2.0)
            ll = np.r_[noaa_station.lon.values, noaa_station.lat.values]
            xy =proj_utils.mapper('WGS84','EPSG:26910')(ll)
            row,col = utils.nearest_2d(met.x, met.y, xy[0], xy[1])
            ax.plot( self.met.time, self.met.air_temperature.isel(row=row,col=col),
                     label=f"{name} old", color=ls[0].get_color(), lw=0.80)
        fig.tight_layout()
        ax.legend(loc='upper left')
        ax.axis((19212.31300939389, 19213.87069539253, 11., 35))
        return fig,ax

    def interpolate_new_temperature(self):
        # interpolate to grid roughly
        noaa_ll = np.c_[self.noaa_air_temp.lon.values, self.noaa_air_temp.lat.values]
        noaa_xy = proj_utils.mapper('WGS84','EPSG:26910')(noaa_ll)
        met_XY = np.stack((self.met.x, self.met.y), axis=-1)
        
        # sort of slow to re-do the interpolation at each step, so get the set of "footprints"
        # once.
        basis_functions = [ field.XYZField(noaa_xy, np.arange(noaa_xy.shape[0])==stn_idx,default_interpolation='nearest')(met_XY)
                            for stn_idx in range(noaa_xy.shape[0]) ]
        
        temp_new = self.met['air_temperature'].values.copy() # (time,row,col)
        temp_new[:,:,:] = 0.0
        
        for stn_idx, basis in enumerate(basis_functions):
            T_on_met_time = utils.interp_near(self.met.time.values, 
                                              self.noaa_air_temp.time.values, 
                                              self.noaa_air_temp.air_temperature.isel(station=stn_idx))
            temp_new[:,:,:] += T_on_met_time[:,None,None] * basis[None,:,:]
        return temp_new
