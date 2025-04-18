from stompy import memoize, utils, xr_utils
from bloom_common import load_model, ratio
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import stompy.plot.cmap as scmap
from matplotlib import cm, colors

disc_viridis=scmap.cmap_discretize(cm.viridis,6)


class TracerGroup:
    tag=None
    run_dir=None
    swim=None
    initial=None
    suffix=None
    initial_release=True # all tracer was added in initial condition.
    temp=True # has temperature
    hor_diff=0.0 # added diffusion
    IsatA=10.0 # Isat used for kRadA tracer
    IsatB=20.0 # Isat used for kRadB tracer

    # cart_fac=1.07**5 # correction for no-temp run in some cases
    cart_fac=1.0 # bug in CART process means no-temp run have 1.07**-5 factor
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
    @property
    def model(self):
        return load_model(self.run_dir)

    @property
    def grid(self):
        return self.ds.grid
        
    @property
    def ds(self):
        return self.model.map_dataset()

    def __getitem__(self,k):
        vname = f'{k}{self.suffix}'
        if vname in self.ds:
            return self.ds[vname]
        elif k=='age' and self.initial_release:
            return (self.ds.time - self.ds.time[0])/np.timedelta64(24,'h')
        elif k in self.ds:
            return self.ds[k]
        else:
            return None
        #else:
        #    raise Exception(f"Can't handle field {k}")

    @property
    def tracer_short_name(self):
        if np.isreal(self.swim):
            swim_str=f"{self.swim:.1f}mpd"
        else:
            swim_str=f"{np.imag(self.swim):.1f}mpd_diurnal"
            
        return f'conc{self.suffix}_{swim_str}_{self.initial}'
        
    def extract_transect(self, linestring, tidx, tracers=['conc']):
        cells,dists = self.profile_cell_dists(linestring)

        tran=xr.Dataset()
        snap = self.ds.isel(time=tidx)
    
        # DFM map output we have to reconstruct sigma...
        tran['LayCoord_cc'] = snap['LayCoord_cc']
        tran['LayCoord_w'] = snap['LayCoord_w']
        # inputs for sigma transform
        tran['s1'] = snap['s1'].isel(nFlowElem=cells)
        tran['FlowElem_bl'] = snap['FlowElem_bl'].isel(nFlowElem=cells)
    
        tran['z_ctr'] = xr_utils.decode_sigma(tran, tran.LayCoord_cc)
        tran['z_int'] = xr_utils.decode_sigma(tran, tran.LayCoord_w)
    
        tran = tran.set_coords(['LayCoord_cc','LayCoord_w','z_ctr','z_int'])
        
        #tran['z_int']=('sample','interface'),interfaces
        #tran['z_ctr']=('sample','layer'), 0.5*(interfaces[:,:-1] + interfaces[:,1:])
    
        for tracer in tracers: 
            #tran[tracer] = snap[tracer].isel(nFlowElem=cells)
            tran[tracer] = self[tracer].isel(time=tidx,nFlowElem=cells)

        cc=self.grid.cells_center()
        tran = tran.rename({'nFlowElem':'sample', 'laydim':'layer', 'wdim':'interface'})
        tran['d_sample']=('sample',),dists
        tran['cell']=('sample',),cells
        tran['x_sample']=cc[cells,0]
        tran['y_sample']=cc[cells,1]
    
        return tran

    @memoize.imemoize(lru=10)
    def extract_tracers(self,tidx,layer,Isat=10.0,thresh=1e-5, light_lim='mean_lim'):
        # Extraction
        # instantaneous release, so age is a given.
        t=self.ds.time.isel(time=tidx).values
        age_d = (t - self.ds.time.values[0]) / np.timedelta64(86400,'s')

        assert layer!='mean'

        snap=self.ds.isel(time=tidx,laydim=layer)
        conc  = snap[f'conc{self.suffix}'].values

        if light_lim=='lim_mean':
            # average irradiance accounting for Kd, vertical mixing.
            Imean = ratio( self.cart_fac*snap[f'radc{self.suffix}'].values, conc*age_d, thresh)
            kLight = Imean/(Imean + Isat)
        elif light_lim=='mean_lim':
            Imean = None
            if Isat==self.IsatA:
                kLight = ratio( self.cart_fac * snap[f'kRadA{self.suffix}'].values, conc*age_d, thresh)
            elif Isat==self.IsatB:
                kLight = ratio( self.cart_fac*snap[f'kRadB{self.suffix}'].values, conc*age_d, thresh)
            elif Isat==0.0:
                kLight = np.ones_like(conc)
            else:            
                kLightA = ratio( self.cart_fac * snap[f'kRadA{self.suffix}'].values, conc*age_d, thresh)
                kLightB = ratio( self.cart_fac * snap[f'kRadB{self.suffix}'].values, conc*age_d, thresh)

                # Fit the line directly
                log_kLightA = np.log(kLightA.clip(1e-5,1.0))
                log_kLightB = np.log(kLightB.clip(1e-5,1.0))
                log_kLight_slope = (log_kLightB - log_kLightA) / (self.IsatB - self.IsatA)
                # m*IsatA+b=log_fA
                log_kLight0 = log_kLightA - log_kLight_slope*self.IsatA
                kLight0 = np.exp(log_kLight0)
                kLight = kLight0*np.exp(log_kLight_slope*Isat)

        #Imean[np.isnan(Imean)]=0.0 
        #kLight = fill(kLight, iterations=120)
        #print(f"kLight: {np.isnan(kLight).sum()} missing values, thresh={thresh}")
        return dict(age_d=age_d, conc=conc, kLight=kLight, Imean=Imean, t=t)  
    
    @memoize.imemoize()
    def profile_cell_dists(self,linestring):
        return self.grid.select_cells_intersecting(linestring,order=True,return_distance=True)
        

    def figure_fields(self,tidx,fields=['logConc','meanRad','fMeanRad','meanfRadA'],
                     layer='Surface', use_abs=True, Isat=10.0,
                     zoom=(544806.72535673, 584499.458477138, 4145536.190742454, 4197985.175969432)
                     ):

        if layer=='Surface':
            laydim=-1
        else:
            laydim=layer

        # First, the basic light limitation calculation:
        sel=dict(time=tidx,laydim=laydim)

        def maybe_abs(x):
            # DFM gets some negative concentrations. Slightly better to take abs of everything.
            # For lagrangian model doesn't matter too much since concentration is quite low.
            if use_abs: return np.abs(x)
            else: return x

        @memoize.memoize()
        def d(field):
            if field=='conc':
                return maybe_abs(self['conc'].isel(**sel).values)
            elif field=='radc':
                return maybe_abs(self['radc'].isel(**sel).values)
            elif field=='kRadA':
                return maybe_abs(self['kRadA'].isel(**sel).values)
            elif field=='kRadB':
                return maybe_abs(self['kRadB'].isel(**sel).values)


        age=((self.ds.time[tidx] - self.ds.time[0])/np.timedelta64(24,'h')).item()

        fig,axs=plt.subplots(1,len(fields),figsize=(11,4.25))
        fig.subplots_adjust(left=0.02,right=0.98,top=0.98,bottom=0.07,wspace=0.05)
        for ax,field in zip(axs,fields):
            ax.set_adjustable('datalim')
            cmap='turbo'
            units=''
            norm=colors.Normalize
            label=field

            if field=='logConc':
                scal=d('conc')
                norm=colors.LogNorm
                clim=[1e-4,10]
                units=r'$\mu$g l$^{-1}$ chl-a'
                label="log$_{10}$ conc"
            elif field=='meanRad':
                scal=ratio(d('radc'),d('conc')*age)
                clim=[0,50]
                units='W m$^{-2}$'
                cmap='inferno'
                label='mean(Rad)'
            elif field=='fMeanRad':
                rad = ratio(d('radc'),d('conc')*age)
                Isat=10.0
                scal=0.3*rad/(rad+Isat)
                clim=[0.1,0.4]
                label='0.3*f$_I$(mean(rad))'
                cmap=disc_viridis #'viridis'
                units='Growth limitation'
            elif field=='meanFRadA':
                scal=ratio(d('kRadA'),age*d('conc'))
                clim=[0.1,0.4]
                label='mean(f$_I$(rad))'
                cmap=disc_viridis #'viridis'
                units='Growth limitation'
            else:
                raise Exception(f"Bad field: {field}")

            ccoll=self.grid.plot_cells(values=scal,cmap=cmap,norm=norm(),
                                       lw=0.5,ec='face',clim=clim,ax=ax)
            plt.colorbar(ccoll,orientation='horizontal',label=units,shrink=0.9,fraction=0.1,pad=0.02)
            ax.text(0.40,0.98,label,
                    fontsize=12,transform=ax.transAxes,va='top')

            ax.axis(zoom)
            ax.xaxis.set_visible(0)
            ax.yaxis.set_visible(0)

        axs[0].text(0.02,0.03,utils.strftime(self.ds.time[tidx].values - np.timedelta64(7,'h'),
                                             "%Y-%m-%d %H:%M PDT"),
                    fontsize=14,transform=axs[0].transAxes)
        return fig

    
# Spinup
#run_dir="bloom_tracers_v01/run_20220801T0000_20220804T0000_v01"
# Main run
#run_dir="bloom_tracers_v01/run_20220804T0000_20220830T0000_v00"

# 2024-06-28: improved temperature field, 3 choices of swimming
#run_dir="bloom_tracers_v09/run_20220804T1820_20220830T0000_v01"
#swim_speeds=[5,10,0]

# 2024-07-10ish: fix bug in initial condition, small and large release from RS.
# Failed 3 days early due to full disk.
#run_dir="bloom_tracers_v10/run_20220804T1820_20220830T0000_v00"
#swim_speeds=[5,10,0,5,10,0]
#initial_conds=['alameda','alameda','alameda','southbay','southbay','southbay']

# no temperature run -- only 5 days in as of 2024-07-23
# run_dir="bloom_tracers_v11/run_20220804T1820_20220830T0000_v00"
# swim_speeds=[5,10,0,5,10,0]
# initial_conds=['alameda','alameda','alameda','southbay','southbay','southbay']

tracer_groups=[]

run_dir="bloom_tracers_v12/run_20220804T1820_20220830T0000_v00"

# 2024-08-30: v12 -- lots of tracers, uniform and chl_from_rs
# v12: lagrangian and uniform, includes diurnal swimming, 80um/day swimming
tracer_groups += [
    TracerGroup(tag='v12',suffix=0,run_dir=run_dir,swim=5,     initial='alameda'),
    TracerGroup(tag='v12',suffix=1,run_dir=run_dir,swim=10,    initial='alameda'),
    TracerGroup(tag='v12',suffix=2,run_dir=run_dir,swim=0,     initial='alameda'),
    TracerGroup(tag='v12',suffix=3,run_dir=run_dir,swim=6.912, initial='alameda'),
    TracerGroup(tag='v12',suffix=4,run_dir=run_dir,swim=6.912j,initial='alameda'),
    TracerGroup(tag='v12',suffix=5,run_dir=run_dir,swim=5,     initial='uniform'),
    TracerGroup(tag='v12',suffix=6,run_dir=run_dir,swim=10,    initial='uniform'),
    TracerGroup(tag='v12',suffix=7,run_dir=run_dir,swim=0,     initial='uniform'),
    TracerGroup(tag='v12',suffix=8,run_dir=run_dir,swim=6.912, initial='uniform'),
    TracerGroup(tag='v12',suffix=9,run_dir=run_dir,swim=6.912j,initial='uniform'),
]    


# v13: lagrangian, diurnal swimming with 80um/day, incl. linearized light
run_dir="bloom_tracers_v13/run_20220804T1820_20220830T0000_v01"
tracer_groups += [
    TracerGroup(tag='v13',suffix=0,run_dir=run_dir,swim=0,initial='alameda'),
    TracerGroup(tag='v13',suffix=1,run_dir=run_dir,swim=6.912,initial='alameda'),
    TracerGroup(tag='v13',suffix=2,run_dir=run_dir,swim=6.912j,initial='alameda'),
    TracerGroup(tag='v13',suffix=3,run_dir=run_dir,swim=0,initial='southbay'),
    TracerGroup(tag='v13',suffix=4,run_dir=run_dir,swim=6.912,initial='southbay'),
    TracerGroup(tag='v13',suffix=5,run_dir=run_dir,swim=6.912j,initial='southbay'),
]

# v14: no-temperature
run_dir="bloom_tracers_v14/run_20220804T1820_20220830T0000_v00"
tracer_groups += [
    TracerGroup(tag='v14', run_dir=run_dir, suffix=0, swim=0,      temp=False, cart_fac=1.07**5, initial='alameda'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=1, swim=6.912,  temp=False, cart_fac=1.07**5, initial='alameda'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=2, swim=6.912j, temp=False, cart_fac=1.07**5, initial='alameda'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=3, swim=0,      temp=False, cart_fac=1.07**5, initial='southbay'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=4, swim=6.912,  temp=False, cart_fac=1.07**5, initial='southbay'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=5, swim=6.912j, temp=False, cart_fac=1.07**5, initial='southbay'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=6, swim=0,      temp=False, cart_fac=1.07**5, initial='uniform'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=7, swim=5.0,    temp=False, cart_fac=1.07**5, initial='uniform'),                        
    TracerGroup(tag='v14', run_dir=run_dir, suffix=8, swim=6.912,  temp=False, cart_fac=1.07**5, initial='uniform'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=9, swim=6.912j, temp=False, cart_fac=1.07**5, initial='uniform'),
    TracerGroup(tag='v14', run_dir=run_dir, suffix=10, swim=10.0,  temp=False, cart_fac=1.07**5, initial='uniform'),                        
]

# v15: diffusion
run_dir="bloom_tracers_v15/run_20220804T1820_20220830T0000_v00"
tracer_groups += [
    TracerGroup(tag='v15', run_dir=run_dir, suffix=0, swim=0,      initial='alameda',  hor_diff=10.0),
    TracerGroup(tag='v15', run_dir=run_dir, suffix=1, swim=6.912,  initial='alameda',  hor_diff=10.0),
    TracerGroup(tag='v15', run_dir=run_dir, suffix=2, swim=6.912j, initial='alameda',  hor_diff=10.0),
    TracerGroup(tag='v15', run_dir=run_dir, suffix=3, swim=0,      initial='southbay', hor_diff=10.0),
    TracerGroup(tag='v15', run_dir=run_dir, suffix=4, swim=6.912,  initial='southbay', hor_diff=10.0),
    TracerGroup(tag='v15', run_dir=run_dir, suffix=5, swim=6.912j, initial='southbay', hor_diff=10.0)
]

# v16: new Kd field
run_dir="bloom_tracers_v16/run_20220804T1820_20220830T0000_v00"
tracer_groups += [
    TracerGroup(tag='v16',suffix=0,run_dir=run_dir,swim=0,     initial='alameda'),
    TracerGroup(tag='v16',suffix=1,run_dir=run_dir,swim=6.912, initial='alameda'),
    TracerGroup(tag='v16',suffix=2,run_dir=run_dir,swim=6.912j,initial='alameda'),
    TracerGroup(tag='v16',suffix=3,run_dir=run_dir,swim=0,     initial='southbay'),
    TracerGroup(tag='v16',suffix=4,run_dir=run_dir,swim=6.912, initial='southbay'),
    TracerGroup(tag='v16',suffix=5,run_dir=run_dir,swim=6.912j,initial='southbay'),
]    
