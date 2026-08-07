#!/usr/bin/env python
# coding: utf-8

# Adding Parameter Optimization
# ==
# 
# DIN supplement, cycling
# 
# Top-down control by zoop, virus
# 
# This version builds on bloom_lagrangian_zoop_etal_tuning, and continues the work on 
# parameter optimization.

import matplotlib.pyplot as plt



#import sys
#s="/richmondvol1/rusty/stompy"
#if s not in sys.path: sys.path.append(s)
    
from importlib import reload
import time
import datetime
import os

import matplotlib.colorbar as cbar
import pandas as pd
import numpy as np
import xarray as xr

from scipy.integrate import solve_ivp
from matplotlib import colors, cm, ticker
import seaborn as sns

from stompy import utils, memoize, xr_utils, xr_transect
import stompy.model.delft.dflow_model as dfm
import stompy.model.delft.waq_scenario as dwaq
from stompy.plot import plot_utils, plot_wkb, mkanim
import stompy.plot.cmap as scmap
from stompy.spatial import field, proj_utils
from stompy.grid import unstructured_grid


# In[8]:

# These are online-coupled dfm-dwaq runs.
import bloom_common
from bloom_common import load_model, ratio, chl_from_RS


# In[9]:

turbo_low = scmap.cmap_transform(cm.turbo,lambda i: i**0.75)

# In[

# Load Supporting Data
# ==

# In[10]:


mooring_ll=pd.read_csv("../Kd_2022/L2/latlon.csv", sep="\t").set_index('station')
mooring_ll


# In[11]:


# Mooring data:
def load_mooring(site='SHL',location_name='Shoal'):
    # site: SHL, SLM, HAY, DMB, SM
    # location_name: match station column in latlon.csv.
    #  "Shoal", "SanLeandroMarina", "Hayward", "Dumbarton Bridge", "San Mateo Bridge"
    mooring_csv=f"../Kd_2022/L2/{site}_all_data_L2.csv"
    df_mooring = pd.read_csv(mooring_csv, low_memory=False)
    del df_mooring['Unnamed: 0']
    df_mooring['time'] = pd.to_datetime(df_mooring['dt'])
    df_mooring['time'] = df_mooring.time.values # should get back to UTC

    # dt has the time zone in it, and .values should get a UTC timestamp, ready for comparison to model output
    sel=(df_mooring.time.values>np.datetime64("2022-08-01")) & (df_mooring.time.values<np.datetime64("2022-09-10"))
    df_mooring = df_mooring.iloc[sel,:] 
    df_mooring = df_mooring[ ['time','Chla_microgL','S_PSU','T_degC','Turb_FNU'] ]
    df_mooring = df_mooring.set_index('time').resample('1H').mean().reset_index()

    # For now just a single mooring, but could be multiple moorings
    df_mooring['site']=site

    lonlat = np.r_[mooring_ll.loc[location_name,'longitude'], mooring_ll.loc[location_name,'latitude']]
    xy = proj_utils.mapper('WGS84','EPSG:26910')(lonlat)
    df_mooring['x']=xy[0]
    df_mooring['y']=xy[1]
    return df_mooring

df_SHL = load_mooring('SHL',"Shoal")
df_SLM = load_mooring('SLM',"SanLeandroMarina")
df_DMB = load_mooring('DMB',"Dumbarton Bridge")
df_HAY = load_mooring('HAY',"Hayward")
df_mooring = df_SHL


# In[12]:


if 0:
    fig,ax = plt.subplots()

    for df in [df_SHL,df_SLM,df_DMB]:
        ax.plot(df.time, df.Chla_microgL, label=df.site.values[0])
    ax.set_ylabel(f"{df_mooring.site.values[0]} chl-a ug/l")
    ax.legend()
    fig.autofmt_xdate()


# In[13]:


# Confirm mooring location
if 0:
    fig,ax=plt.subplots()
    grid=tracer_groups[0].grid
    grid.plot_edges(color='0.6',alpha=0.5,lw=0.5)
    for df in [df_SHL,df_SLM,df_DMB]:
        ax.plot(df.iloc[0].x, df.iloc[0].y,'ro')
        ax.text(df.iloc[0].x, df.iloc[0].y,df.site.values[0])
    ax.axis('off')


# In[14]:


import tracer_group
reload(tracer_group)
from tracer_group import TracerGroup, tracer_groups

# Alameda release is basically zero until after the first pulse of bloom
tg = [t for t in tracer_groups if t.tag=='v17' and t.swim==6.912 and t.initial=='alameda'][0]
# But South Bay release shows up too early for DMB, SLM
#tg = [t for t in tracer_groups if t.tag=='v17' and t.swim==6.912 and t.initial=='southbay'][0]


# In[15]:


grid_poly = tracer_groups[0].grid.boundary_polygon()


# Model Code
# ==

# In[16]:


from stompy import memoize
class LagrangianModel:    
    # Any parameters that are available for tuning go here with defaults
    base_params=dict(
        thresh=1e-2,
        c0=3,
        kprod=1.0,
        N0=35, # initial DIN, uM
        Nsat=0.5, # half-sat DIN, uM
        Z0=0.1, # initial zoop. units?
        kmort=0.05,
        Isat=10.0,
        background=4.0, # baseline chl, added to result after the fact.
        alpha=0.15, # stoichiometry
        mu_Z=0.1, # growth rate for Zoop
    )


    # expects fields in dataframe:
    # x,y -  location of the sample
    # time 
    # optionally observed values, e.g. Chla_microgL 
    df_inputs=None # determines the samples that will be simulated. 
    
    group=None # tracer group
    layer=15 # model layer requires some preprocessing and it's discrete, not available for tuning
    light_lim='mean_lim' # approach to calculating light limitation
    # 'mean_lim', 'lim_mean'. May eventually have 'mean_lim_selfshade'
    # What to report when there's not enough tracer to do anything:
    P_nodata=np.nan
    N_nodata=np.nan 

    min_ageconc = 1e-5 # threshold passed to bloom_common.ratio()
    drop_by_ageconc = False # False: keep inputs even if they will be nan. True: drop them from the inputs. 
    
    def __init__(self,params={},**kw):
        # Makes a copy for this instance. So you can have the "global" defaults defined above,
        # set some "instance" defaults, and still tune some parameters with different values within
        # and instance. 
        self.base_params = self.base_params | params # params will take precedence
        utils.set_keywords(self,kw)
        assert self.df_inputs is not None,"Must specify df_inputs"
        self.prepare_inputs()
        self.parameter_sets=[]
        self.evaluate_parameter_set()

    @memoize.imemoize()
    def xy_to_cell(self,xy):
        return self.group.grid.select_cells_nearest(xy)
            
    def prepare_inputs(self):
        """
        Copy and augment df_inputs with tracer data.
        """
        xy = self.df_inputs[ ['x','y'] ].values
        cells = np.array([self.xy_to_cell(p) for p in xy])
        
        # Some may be -1
        tidx = utils.nearest(self.group.ds.time.values, self.df_inputs.time.values, max_dx = np.timedelta64(1,'h'))
        valid = tidx>=0

        # Filter to samples that are covered by the tracer data
        self.df_inputs = self.df_inputs.loc[ valid, :].copy()
        self.df_inputs['cell'] = cells[valid]
        self.df_inputs['tidx'] = tidx[valid]

        df_data = self.group.extract_timeseries(self.df_inputs.tidx.values, self.df_inputs.cell.values, layer=self.layer)
        for col in df_data.columns:
            self.df_inputs[col] = df_data[col].values
        self.df_inputs['conc'] = self.df_inputs['conc'].clip(0.0)        
    
        if self.drop_by_ageconc:
            ageconc_valid = (self.df_inputs['conc']*self.df_inputs['age_d']).values >= self.min_ageconc
            self.df_inputs = self.df_inputs.loc[ageconc_valid, :].copy() 
        
    def evaluate_parameter_set(self,**kw):
        """
        Runs the forward model on all samples in self.df_inputs, 
        using any specified parameters, falling back to base_params.
        Returns dataframe with the results.
        kw can include any fields from base_params.
        """
        t0=time.time()
        # params that are set across an evaluation of all the data points
        params=self.base_params | kw
        self.parameter_sets.append(params) # or just log kw
            
        # ODE integration
        # State vector is [N, P, Z, I, V]
    
        recs=[]
        
        for i,row in utils.progress(enumerate(self.df_inputs.itertuples()),msg="Evaluate samples: %s"):
            constant_params=dict(params) # make a copy so code in this loop can use df_inputs to modify parameters. 
            IC=np.r_[params['N0'], params['c0']*row.conc, params['Z0'], params['I0'], params['V0']]
            age_d=row.age_d
            # self-shading would probably mean kLight is at least partially calculated inside the
            # integration
            constant_params['kLight']=bloom_common.calc_kLight(Isat=constant_params['Isat'],
                                                               kRadA=row.kRadA,IsatA=row.IsatA,
                                                               kRadB=row.kRadB,IsatB=row.IsatB,
                                                               conc=row.conc,age_d=row.age_d)
            if np.isnan(constant_params['kLight']):
                result = np.nan*IC
            elif age_d>0:
                bunch = solve_ivp(self.npziv_diff, y0=IC, t_span=[0,row.age_d], args=(constant_params,))
                result=bunch.y[:,-1]
            else:
                result = IC
            recs.append(dict(N=result[0],
                             P=result[1]+constant_params['background'],
                             Z=result[2],
                             I=result[3],
                             V=result[4],
                             time=row.time)
                        | constant_params)
        self.df_result=pd.DataFrame(recs)
    
        elapsed=time.time() - t0
        param_str=",".join( [f'{k}={params[k]:.3f}' for k in params] )
        #print(f"solveNPZIV({param_str}) in {elapsed:.3f}s")

    # Can override in subclass
    def npziv_diff(self, t, state, params):
        N,P,Z,I,V=np.clip(state,0,None)

        kDIN    =  N/(N+params['Nsat'])
        dgrossP =  params['kprod']*params['kLight']*kDIN*P
        dnetP   = -params['kmort']*P + dgrossP
        dN      = -params['alpha']*dgrossP
        dZ      =  params['mu_Z']*Z
        # default derivative ignores infected and virus state
        return np.r_[dN,dnetP,dZ,0,0]

    @property
    def df_merged(self):
        return pd.concat([self.df_result, self.df_inputs.reset_index()], axis=1)
        
def fig_time_series(model, df_result=None, mask=None):
    if df_result is None:
        df_result = model.df_result
    fig,axs = plt.subplots(4,1,sharex=True,figsize=(7,6.5))

    df_inputs=model.df_inputs

    if mask is not None:
        df_inputs = df_inputs.loc[mask,:]
        df_result = df_result.loc[mask,:]
        
    ax=axs[0]
    site = df_inputs.site.values[0]
    #ax.plot(model.df_observations.time, model.df_mooring.Chla_microgL, label=f'{site} mooring')
    ax.plot(df_inputs.time, df_inputs.Chla_microgL, label=f'{site} mooring')
    ax.plot(df_result.time, df_result.P, label='Lagrangian')
    ax.set_ylabel('Chl-a (ug/l)')
    ax.legend(loc='upper left')
    ax.axis(ymin=0)
    
    ax=axs[1]
    ax.plot(df_result.time, df_result.N)
    ax.set_ylabel('N ($\\mu$M)',color=ax.lines[0].get_color())
    ax.axis(ymin=-0.5)
    ax=ax.twinx()
    ax.plot(df_result.time, df_result.Z,color='tab:orange')
    ax.set_ylabel('Z (-)',color='tab:orange')
    
    ax=axs[2]
    ax.plot(df_inputs.time, df_inputs.conc)
    ax.set_ylabel('Tracer conc')
    ax.axis(ymin=-0.025)
    
    ax=axs[3]
    ax.plot(df_result.time, df_result.kLight)
    ax.set_ylabel('kLight (-)')
    ax.axis(ymin=0)
    
    fig.autofmt_xdate()
    fig.subplots_adjust(top=0.98,bottom=0.1,left=0.1,right=0.87,hspace=0.07)
    return fig

def fig_time_series_with_virus(model, df_result=None, mask=None):
    if df_result is None:
        df_result = model.df_result
    fig,axs = plt.subplots(4,1,sharex=True,figsize=(7,6.5))

    df_inputs = model.df_inputs

    if mask is not None:
        df_inputs = df_inputs.loc[mask,:]
        df_result = df_result.loc[mask,:]

    ax=axs[0]
    site = df_inputs.site.values[0]
    # Note - df_inputs is already decimated, but the corresponding df_mooring is not. Could include as optional var
    #ax.plot(model.df_mooring.time, model.df_mooring.Chla_microgL,label=f'{site} mooring')
    ax.plot(df_inputs.time, df_inputs.Chla_microgL, label=f'{site} mooring (sampled)')
    ax.plot(df_result.time, df_result.P, label='Lagrangian')
    ax.set_ylabel('Chl-a (ug/l)')
    ax.legend(loc='upper left')
    if 1: # drop outliers from limits calc
        ymax = max(df_inputs.Chla_microgL.quantile(0.995),
                   df_result.P.max())
        ax.axis(ymin=0,ymax=ymax)
    
    ax=axs[1]
    ax.plot(df_result.time, df_result.N)
    ax.set_ylabel('N ($\\mu$M)',color=ax.lines[0].get_color())
    ax.axis(ymin=-0.5)
    if np.any(df_result.Z>0.0):
        ax=ax.twinx()
        ax.plot(df_result.time, df_result.Z,color='tab:orange')
        ax.set_ylabel('Z (-)',color=ax.lines[0].get_color())
        plt.setp(ax.get_yticklabels(), color=ax.lines[0].get_color())
        ax.axis(ymin=-0.01)

    has_virus=np.any(df_result.I.values>0)

    if has_virus:
        ax=axs[2]
        ax.plot(df_result.time, df_result.I, color='tab:red')
        ax.set_ylabel('Infected (ug chl/l)',color=ax.lines[0].get_color())
        plt.setp(ax.get_yticklabels(), color=ax.lines[0].get_color())
        ax.axis(ymin=0)
        ax=ax.twinx()
        ax.plot(df_result.time, df_result.V, color='tab:orange')
        ax.set_ylabel('Virus ($\\sim$VLP mL$^{-1}$)',color=ax.lines[0].get_color())
        plt.setp(ax.get_yticklabels(), color=ax.lines[0].get_color())
        ax.axis(ymin=0)
    else:
        axs[2].set_visible(0)

    ax=axs[3]
    ax.plot(df_inputs.time, df_inputs.conc)
    ax.set_ylabel('Tracer conc')
    ax.axis(ymin=-0.025)
    
    fig.autofmt_xdate()
    fig.subplots_adjust(top=0.98,bottom=0.1,left=0.13,right=0.87,hspace=0.07)
    return fig


# handle types of input by composition, with prepare* and plot* methods
# single mooring, multiple mooring, single RS scene, multiple RS scene.
def prepare_mooring_input(df_mooring,stride=1):
    return df_mooring.iloc[::stride].copy()

def prepare_moorings_input(dfs,stride=1):
    dfs=[prepare_mooring_input(df,stride) for df in dfs]
    return pd.concat(dfs)
    


# In[17]:


class NPZIVModel(LagrangianModel):
    base_params=dict(
        N0=35, # initial DIN, uM
        # P0=c0*tracer conc
        Z0=0.1, # initial zoop. units?
        I0=0.0,
        V0=0.0,
        c0=3, # scale tracer output to get initial concentration
        kprod=1.0, # maximum instantaneous phyto growth rate [1/d]
        Nsat=0.5, # half-sat DIN, uM
        Isat=10.0, # half-sat for light, in some version(?) of W/m2
        kmortP=0.05, # phyto mortality rate [1/d]
        kmortZ=0.2, # zooplankton mortality rate [1/d]
        background=2.0, # baseline chl, added to result after the fact.
        alpha=0.15, # stoichiometry
        Z_efficiency=0.3, # growth efficiency for zoop,
        Z_max_ration=1.5, # maximum ration for zooplankton [1/d]
        mortP_to_N_efficiency = 0.5,
        N_supplement = 1.0, # uM/day
        thresh=1e-2,
        Ivlev=1.0, # Ivlev constant, 1.0  [ 1/(uM N) ]
        k_infect=0.0, # or 6.75e-8 [mL VLP^-1 day^-1] (adjust based on assumed units for P and V)default=5e-9
        lambda_lysis=1.5, # (float): Lysis rate of infected phytoplankton [day^-1]
        B=500, # viruses per 'host unit'; default=500
        k_virus=0.25, # Viral decay/inactivation rate [day^-1]
    )

    def npziv_diff(self, t, state, params):
        N,P,Z,I,V=np.clip(state,0,None)
        
        lim_DIN=N/(N+params['Nsat'])

        # No self-shading yet. 
        dgrossP = params['kprod']*params['kLight']*lim_DIN*P
        dmortP = params['kmortP']*P # positive value, but represents loss of P
        # Is this legit? Wikipedia says it is
        dgrazeP = Z*params['Z_max_ration']*(1-np.exp(-params['Ivlev']*P))

        # Virus
        # terms from basic_ode_phyto_graze_virus3.py
        # dPdt = mu_max /Kd_p * P - g * P - k * P * V
        # dIdt = k * P * V - lambda_lysis * I
        # dVdt = B * lambda_lysis * I - k * P * V - d_V * V
        # k (float): Viral infection rate constant (mL VLP^-1 day^-1).
        dinfectP = params['k_infect']*P*V
        
        dnetP = dgrossP - dmortP - dgrazeP - dinfectP

        dlyseI =  params['lambda_lysis'] * I
        
        # From DS equations, phytoplankton, when infected, are removed from P and added to I
        # at which point they lyse at a first order rate, and have no growth.
        dZ = params['Z_efficiency'] * dgrazeP - params['kmortZ']*Z
        # Wikipedia NPZ has 100% recycling efficiency for P and Z, including a (1-Z_efficiency) term
        # for "sloppy grazing"
        dN = (
            - params['alpha']*dgrossP # uptake
            + params['mortP_to_N_efficiency']*params['alpha']*dmortP # recycling
            + params['alpha'] * (1-params['Z_efficiency'])*dgrazeP
            + params['N_supplement'] # uM/day continual source
        )
        
        dI=dinfectP - dlyseI
        dV=params['B']*dlyseI - dinfectP - params['k_virus'] * V
        return np.r_[dN,dnetP,dZ,dI,dV]


# Manual Tests
# ==

# In[18]:


if 0:
    # An okay fit without virus.
    params=dict(c0=6, I0=0, N_supplement=0.25, N0=10, kprod=1.5,Isat=5,
                Z_max_ration=2.1, Z0=0.0003)
    for df in [df_HAY]: #,df_DMB,df_SLM]:
        model=NPZIVModel(group=tg,
                         df_inputs=prepare_mooring_input(df, stride=5),
                         params=params, layer='mean')
        fig_time_series_with_virus(model)
    
    None


# In[19]:


# Testing virus
# original starting point: 
# k_infect=8e-7  B=20_000 k_virus=0.25, lambda_lysis=4.0
# starting point based on DS email 9/30/25:
#  k_infect=1e-5, B=900, k_virus=0.025 lambda_lysis=1.50
if 0:
    # DMB is better with kprod=1.7 -- 1.8
    # 
    params=dict(N0=10, c0=5, Z0=0.0000, I0=1e-4, V0=200,
                # virus params
                k_infect=1.00e-5, B=900, k_virus=0.025, lambda_lysis=1.5,
                # other params
                mortP_to_N_efficiency=0.8, N_supplement=1.5, kprod=1.6,
                # Can Isat help with SHL growing too fast? sort of
                Isat=20.0)
    for df in [df_SHL,df_DMB]: #,df_DMB,df_SLM]:
        model=NPZIVModel(group=tg,
                         df_inputs=prepare_mooring_input(df, stride=5),
                         params=params, layer='mean')
        
        fig_time_series_with_virus(model)
    
    None


# Spatial Results
# ==

# In[20]:


@memoize.memoize(lru=10)
def get_fill_func(grid,cells,count):
    return grid.fill_by_smoothing_func(cells, count=count) # 10s
   


# In[21]:


def fig_spatial(model,df_result=None,fig=None,fill_count=45):
    # fill_count should be about 3x stride
    if df_result is None:
        df_result=model.df_result
    chl_cmap='turbo'
    chl_clim=[0,400]
    #chl_norm=colors.PowerNorm(0.4,*chl_clim)
    chl_norm=colors.LogNorm(chl_clim[1]*0.05, chl_clim[1], True)

    t=model.df_inputs['time'].values[0] # Assumes snapshot, all same time

    if fig is None:
        fig = plt.figure(figsize=(7.5,5))
    else:
        fig.clf()
    axs = [fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)]
    fig.subplots_adjust( wspace=0.04,top=0.97,right=0.97,left=0.03)

    for ax in axs: 
        ax.set_adjustable('box')
    
    ax=axs[0] # model results
    # Could cache fill_func on self
    P_result = df_result.P.values
    P_result = np.where( np.isfinite(P_result), P_result, 0.0) # fill in with 0.0
    fill_func = get_fill_func(model.group.grid,model.df_inputs.cell.values, count=fill_count) # 10s or 0.06s cached.
    
    P_filled = fill_func(P_result)
    coll=model.group.grid.plot_cells(values=P_filled, cmap=chl_cmap, norm=chl_norm, 
                                     lw=0.7,ec='face',ax=ax)
    if 1: # colorbar
        t_str = utils.strftime(t)
        cbar = plt.colorbar(coll,label="Chl-a ($\mu g \ l^{-1}$)\n"+t_str,ax=axs,orientation='horizontal',
                           pad=0.04,fraction=0.09,aspect=30)
        cax = cbar.ax
        cax.xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
        cax.xaxis.set_minor_formatter(ticker.FormatStrFormatter('%d'))
        cax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    ax=axs[1] # RS data
    chl_rs = bloom_common.chl_from_RS(t,grid_poly)
    if chl_rs is not None:
        chl_rs.plot(ax=ax, cmap=chl_cmap, norm=chl_norm)
    
    for ax,label in zip(axs,['Lagrangian Model','Imagery']):
        ax.text(0.95,0.95,label,transform=ax.transAxes,ha='right',va='top')
    
    for ax in axs:
        ax.axis(zoom)
        plot_wkb.plot_wkb(grid_poly,lw=0.5,color='k',alpha=0.5,zorder=2, fc='none',ax=ax)
        ax.xaxis.set_visible(0)
        ax.yaxis.set_visible(0)
    return fig


# In[22]:


# construct dataframe of grid locations
zoom=(547000., 591059., 4139804., 4189355.)
stride=15 # only evaluate every n'th cell
grid_poly = tg.grid.boundary_polygon() # for clipping RS data


# In[23]:


def prepare_spatial_input(tg, tidx, zoom, stride=15):
    """
    Spatial sample for plotting -- does not include RS data
    Handles a single snapshot
    """
    # was df_grid_for_time_idx(tidx):
    t = tg.ds.time.values[tidx] 
    cells = np.nonzero(tg.grid.cell_clip_mask(zoom))[0]
    xy = tg.grid.cells_centroid(cells)    
    df_grid = pd.DataFrame()
    df_grid['cell'] = cells
    df_grid['x'] = xy[:,0]
    df_grid['y'] = xy[:,1]
    df_grid['time'] = t
    df_grid = df_grid.iloc[::stride] # solve for only a subset, then interpolate
    return df_grid

def prepare_spatial_with_RS_input(tg, tidx, zoom, stride=15, valid_range=[1.0,500.0]):
    # Expects tidx to correspond to time tg.ds.time.values[tidx] that resolves to a day with
    # good data. Note that RS data 
    df_grid = prepare_spatial_input(tg, tidx, zoom, stride=stride)
    t = tg.ds.time.values[tidx] 
    # => stompy.spatial.field.SimpleGrid
    chl_rs = bloom_common.chl_from_RS(t,tg.grid.boundary_polygon())
    assert chl_rs is not None
    xy = df_grid[ ['x','y'] ].values
    rs_data=chl_rs(xy)
    # assuming the goal is optimizing, go ahead and filter on valid data, only.
    # RS data sometimes comes in with crazy values like 20000
    valid = np.isfinite(rs_data) & (rs_data>=valid_range[0]) & (rs_data<=valid_range[1])
    df_grid = df_grid.iloc[valid,:].copy()
    df_grid['Chla_microgL'] = rs_data[valid]
    return df_grid


# In[24]:


# Dev/test tuning to RS data
# virus
if 0:
    params=dict(N0=10, c0=5, Z0=0.0000, I0=1e-4, V0=200,
                # virus params
                k_infect=1.00e-5, B=900, k_virus=0.025, lambda_lysis=1.5,
                # other params
                mortP_to_N_efficiency=0.8, N_supplement=1.5, kprod=1.6,
                # Can Isat help with SHL growing too fast? sort of
                Isat=20.0)
    model=NPZIVModel(group=tg, 
                     df_inputs=prepare_spatial_with_RS_input(tg,tidx=534,zoom=zoom),
                     params=params)


# In[25]:


#len(model.df_result) # 1169 samples
if 0:
    fig,ax=plt.subplots()
    ax.scatter(model.df_inputs['Chla_microgL'], model.df_result['P'],4)


# In[26]:


# One-shot run the Lagrangian calcs for a snapshot in time, given set of 
# parameters, and plot spatial vs RS chl.
if 0:
    if 1: # virus
        params=dict(N0=10, c0=5, Z0=0.0000, I0=1e-4, V0=200,
                    # virus params
                    k_infect=1.00e-5, B=900, k_virus=0.025, lambda_lysis=1.5,
                    # other params
                    mortP_to_N_efficiency=0.8, N_supplement=1.5, kprod=1.6,
                    # Can Isat help with SHL growing too fast? sort of
                    Isat=20.0)
    if 0: # basic zoop run
        params=dict(c0=6,I0=0, N_supplement=0.25, N0=10, kprod=1.5,Isat=5,
                    Z_max_ration=2.1, Z0=0.0003)
    model=NPZIVModel(group=tg, 
                     df_inputs=prepare_spatial_input(tg,tidx=534,zoom=zoom),
                     params=params)
    
    fig_spatial(model) ;


# Optimization
# ==
# 

# Optimization over Mooring Data
# --

# In[27]:


# Prepare input across multiple moorings:
df_inputss=[]
for df_mooring in [df_SHL,df_DMB,df_SLM,df_HAY]:
    df_inputss.append( prepare_mooring_input(df_mooring,stride=5) )
df_inputs=pd.concat(df_inputss)


# In[28]:


def loss(model):
    P_min=2.0 #
    P_model = model.df_result['P'].values
    P_obs = model.df_inputs['Chla_microgL'].values
    nom_err = (P_model - P_obs)/(P_obs+P_min)
    return np.mean(nom_err**2) # leave as quadratic?

def cost(param_vec, model, params_to_tune):
    param_kw = { p[0]:v for p,v in zip(params_to_tune,param_vec)}
    model.evaluate_parameter_set(**param_kw)
    result = loss(model)
    print(f"{result:.4f} {param_kw}")
    return result

from scipy.optimize import minimize


# In[29]:


# First go, zooplankton: 

# Use powell - much better than default method
# 0.3792 {'c0': 4.506984204870644, 'kprod': 1.1243508317602493, 'N_supplement': 0.16631002953092275, 'N0': 9.931855274572516,
#         'Z0': 0.0002972396053085654, 'Z_max_ration': 2.0980713339308092, 'kmortP': 0.053263070972951444, 
#         'Isat': 3.697702281261708, 'background': 1.6387620761482578, 'Nsat': 0.46703945899618593}
if 0:
    params=dict(c0=6, I0=0, N_supplement=0.25, N0=10, kprod=1.5,Isat=5,
                Z_max_ration=2.1, Z0=0.0003)
    
    model=NPZIVModel(group=tg,
                     df_inputs=df_inputs,
                     params=params, layer='mean',
                     drop_by_ageconc=True)
    
    # Which parameters to tune and their bounds
    params_to_tune=[('c0',0.5,5),
                    ('kprod',0.5,2.0),
                    ('N_supplement',0.05,2),
                    ('N0',5,40),
                    ('Z0',0.00001,0.001),
                    ('Z_max_ration',0.1,4.0), 
                    ('kmortP',0.01,0.5),
                    ('Isat',1.0,80),
                    ('background',0.5,15.0),
                    ('Nsat',0.1,5.0)]
    
    minimize(cost, 
             #x0=[model.base_params[p[0]] for p in params_to_tune],
             # parameters from one round of powell:
             x0=x0_from_last,
             bounds=[tuple(p[1:]) for p in params_to_tune],
             method='powell',
             args=(model,params_to_tune))


# In[31]:


# And a virus attempt

# 0.4947 {'c0': 4.292456597144052, 'kprod': 1.8659785145482262, 'N_supplement': 1.998629843997263, 
#  'N0': 39.99968282526582, 'kmortP': 0.14911867256301087, 'Isat': 12.750108267646562, 'background': 1.691588816926223,
#  'I0': 0.000946428587587386, 'V0': 40.584309054752154, 'k_infect': 1.971633455125515e-05, 'B': 1703.0183485294083,
#  'k_virus': 0.01000010108669568, 'lambda_lysis': 1.150200605580346, 'Nsat': 0.10004313477521526}


# 0.4943 {'c0': 4.241721429774956, 'kprod': 1.8659776468460159, 'N_supplement': 1.9980255522580772, 
#  'N0': 39.996686684477005, 'kmortP': 0.14912122951589343, 'Isat': 12.747880881095213, 'background': 1.6971335926693225,
#  'I0': 0.0007616487613692749, 'V0': 47.34784341126002, 'k_infect': 1.971633455125515e-05, 'B': 1724.6395557772746, 
#  'k_virus': 0.010038691080696379, 'lambda_lysis': 1.0694164723003414, 'Nsat': 0.10004661637908206}

if 1:
    params=dict(c0=4.2417, N0=39.99, kmortP=0.149, 
                Z0=0.0000, # No zooplankton 
                I0=7.616e-4, V0=47.348, # ad hoc virus
                # virus params
                k_infect=1.972e-5, B=1725, k_virus=0.01004, lambda_lysis=1.07,
                # other params
                mortP_to_N_efficiency=0.8, N_supplement=1.999, kprod=1.8660,
                Nsat=0.10005, background=1.697,
                Isat=12.748)
    
    model=NPZIVModel(group=tg,
                     df_inputs=df_inputs,
                     params=params, layer='mean',
                     drop_by_ageconc=True)

    print("Initial guess: ",loss(model))

if 0:
    # Which parameters to tune and their bounds
    params_to_tune=[('c0',0.5,5),
                    ('kprod',0.5,2.0),
                    ('N_supplement',0.05,2),
                    ('N0',5,40),
                    ('kmortP',0.01,0.5),
                    ('Isat',1.0,80),
                    #('background',0.5,15.0),
                    ('I0',0.1e-4,10e-4),
                    ('V0',20,2000),
                    ('k_infect',0.1e-5,5e-5),
                    ('B',100,2000),
                    ('k_virus',0.01,0.10),
                    ('lambda_lysis',0.1, 3.0),
                    # ignore mortP_to_N_efficiency
                    ('Nsat',0.1,5.0)]
    
    minimize(cost, 
             x0=[model.base_params[p[0]] for p in params_to_tune],
             # parameters from one round of powell:
             #x0=x0_from_last,
             bounds=[tuple(p[1:]) for p in params_to_tune],
             method='powell',
             args=(model,params_to_tune))


# In[154]:


x0_from_last = [model.parameter_sets[-1][p[0]] for p in params_to_tune]
#x0_from_last


# In[170]:


model.evaluate_parameter_set(
    **{'c0': 4.29, 'kprod': 1.8659785145482262, 'N_supplement': 1.998629843997263, 
       'N0': 39.99968282526582, 'kmortP': 0.14911867256301087, 'Isat': 12.750108267646562,
       'background': 1.691588816926223, 'I0': 0.000946428587587386, 'V0': 40.584309054752154, 
       'k_infect': 1.971633455125515e-05, 'B': 1703.0183485294083, 'k_virus': 0.01000010108669568,
       'lambda_lysis': 1.150200605580346, 'Nsat': 0.10004313477521526}
)
loss(model) # 0.49467


# In[162]:


x0_from_last


# In[32]:


fig,ax=plt.subplots()

sns.scatterplot(data=model.df_merged, x='Chla_microgL', y='P', hue='site')


# In[34]:


if model.parameter_sets[-1]['k_infect'] > 0.0:
    fig_fn = fig_time_series_with_virus
else:
    fig_fn = fig_time_series 
    
fig_fn(model, mask = model.df_inputs.site.values=='SHL')
fig_fn(model, mask = model.df_inputs.site.values=='SLM')
fig_fn(model, mask = model.df_inputs.site.values=='DMB')
fig_fn(model, mask = model.df_inputs.site.values=='HAY')

("")


# STAN
# ==

# In[47]:


# Assumes a model has been setup and run at least once above
model.df_inputs.to_csv(f"stan/inputs-{tg.tag}-{tg.tracer_short_name}.csv",index=False)


# In[ ]:


functions {
  vector my_ode(real t,        // Current time
                vector y,      // Current state variables
                tuple<Nsat, kprod, kLight, kmortP, Z_max_ration, IvLev, k_infect, lambda_lysis, Z_efficiency, kmortZ,
                      alpha, mortP_to_N_efficiency, k_virus, B> params) { 
    // Parameters (can also be an array/tuple if multiple)
    vector[5] dydt;       // Match dimensions to your system
    
    // Example: Simple exponential decay/growth
    //dydt[1] = theta * y[1]; 

    #def npziv_diff(self, t, state, params):
    real N = y[0];
    real P = y[1];
    real Z = y[2];
    real I = y[3];
    real V = y[4];

    // N,P,Z,I,V=np.clip(state,0,None)
        
    real lim_DIN=N/(N+params['Nsat'])

        # No self-shading yet. 
        dgrossP = params['kprod']*params['kLight']*lim_DIN*P
        dmortP = params['kmortP']*P # positive value, but represents loss of P
        # Is this legit? Wikipedia says it is
        dgrazeP = Z*params['Z_max_ration']*(1-np.exp(-params['Ivlev']*P))

        # Virus
        # terms from basic_ode_phyto_graze_virus3.py
        # dPdt = mu_max /Kd_p * P - g * P - k * P * V
        # dIdt = k * P * V - lambda_lysis * I
        # dVdt = B * lambda_lysis * I - k * P * V - d_V * V
        # k (float): Viral infection rate constant (mL VLP^-1 day^-1).
        dinfectP = params['k_infect']*P*V
        
        dnetP = dgrossP - dmortP - dgrazeP - dinfectP

        dlyseI =  params['lambda_lysis'] * I
        
        # From DS equations, phytoplankton, when infected, are removed from P and added to I
        # at which point they lyse at a first order rate, and have no growth.
        dZ = params['Z_efficiency'] * dgrazeP - params['kmortZ']*Z
        # Wikipedia NPZ has 100% recycling efficiency for P and Z, including a (1-Z_efficiency) term
        # for "sloppy grazing"
        dN = (
            - params['alpha']*dgrossP # uptake
            + params['mortP_to_N_efficiency']*params['alpha']*dmortP # recycling
            + params['alpha'] * (1-params['Z_efficiency'])*dgrazeP
            + params['N_supplement'] # uM/day continual source
        )
        
        dI=dinfectP - dlyseI
        dV=params['B']*dlyseI - dinfectP - params['k_virus'] * V
        return np.r_[dN,dnetP,dZ,dI,dV]



    return dydt;
  }
}

data
{
}

transformed data
{
}

parameters
{
}

transformed parameters
{
}

model
{
}

generated quantities
{
}


# Animation Output
# ==

# In[ ]:


# Render animation
frame_dir = "frames-bloom_lagrangian_zoop_etal-v002"
force=True # overwrite existing images, but does not proactively delete existing.

if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)
    
with open(os.path.join(frame_dir,'params.txt'),'wt') as fp:
    fp.write("\n".join([
        "Parameters:",
        str(params),
        f"Run dir: {tg.run_dir}",
        f"Tracer suffix: {tg.suffix}",
        f"Swim speed: {tg.swim}",
        f"Initial condition: {tg.initial}",
       ]))
    
times=tg.ds.time.values # all times available in the tracer run
tidxs = np.arange(len(times)) # 
#time_sel = slice(None,None,72) # start idx, stop idx, stride of 3 days
time_sel = slice(None,None) # hourly, start to finish
fig=None
for frame,(tidx,t) in enumerate(zip( tidxs[time_sel], times[time_sel])):
    img_fn=os.path.join(frame_dir,f"frame{frame:04d}.png")
    if os.path.exists(img_fn) and not force:
        print("skipping existing image")
        continue
    df_grid_tidx=df_grid_for_time_idx(tidx)
    tuner=NPZIVModel(group=tg, df_mooring=df_grid_tidx, stride=1, params=params)
    fig = tuner.fig_last_eval_spatial(fig=None)
    fig.savefig(img_fn,dpi=100)


# In[186]:


# Compile frames into movie
movie_out=os.path.join(frame_dir,'output.mp4')
mkanim.mkanim(movie_out,frame_dir+"/frame%04d.png")


# In[191]:


# Display resulting animation
Video(movie_out)


# NPZIV Notes
# ==
# 
# 

# In[87]:


# Stoichiometry for DIN vs chl
alpha1 = 6.625 # = 1mol N * 106molC/16 mol N = mol C ; 1mol CO2 fixed * 1mol O2 / 1mol CO2 , 1:1
c_chl = 1/0.03
alpha2 = alpha1*12/c_chl # = 2.385 ug chl / umol N ....106 umol C / 16 umol N * 12 ug C / umol C * 0.03 ug chl / 1 ug
alpha2
1.0/alpha2 # uM DIN / ug/l Chl-a


# Diagnostics for tidal phase
# ==

# In[215]:


# Is there enough salt to diagnose phase? yes - and phase appears correct.
fig,axs=plt.subplots(4,1,sharex=True)
axs[0].plot( df_mooring.time, df_mooring.S_PSU, label='Mooring salinity')

axs[1].plot( tg.ds.time.values, tg.ds['sa1'].isel(laydim=15,nFlowElem=16145),
           label='DFM salinity')

axs[2].plot( df_mooring.time, df_mooring.Chla_microgL,label='Mooring chl-a')

axs[3].plot( tg.ds.time.values, tg.ds[f'conc{tg.suffix}'].isel(laydim=15,nFlowElem=16145),
           label='DWAQ conc.')

for ax in axs:
    ax.legend(loc='upper right', frameon=0)
    
fig.autofmt_xdate()
axs[0].axis((19220.7738, 19222.7409, 32.38, 33.43))
axs[1].axis(ymin=31.35,ymax=31.52)
axs[2].axis(ymin=0,ymax=35)
axs[3].axis(ymin=0,ymax=0.5)
fig.subplots_adjust(right=0.98,bottom=0.13,top=0.98,left=0.10)


# Vertical mixing of P vs N
# ==

# Most of the Lagrangian analysis to date has considered a single layer in isolation.
# 
# In reality, the vertical distribution is dynamic, and we're more interested in the
# water-column averaged growth, optionally transformed to reflect the surface concentration
# for comparison to RS data.
# 
# The integration expects initial values for $N$, $P$, $Z$, $I$, and $V$, with $P$ 
# based on tracer results. Most of these are unchanged since they already represent
# an average concentration with no notion of vertical variability. The one change is
# $P$, which should be the vertical average, including the variable layer thicknesses.
# 
# Tracer results additionally provide $k_{light}$, based on light tracers. This is 
# currently calculated from the local kRadA/kRadB,IsatA,IsatB values.
# 
# I think the approach is then to calculate kLight for each layer, and then combine
# with a biomass-weighted average. There is some room for misinterpretation here -
# it will be biomass weighted average before any growth.  An alternative would
# be to evolve each layer as a separate state variable. Sounds slow. We're assuming
# that things mix tidally.
# 
# Think through some mental experiments:
# 
#  - what happens if there is no swimming, all P is evenly distributed in the vertical?
#    Physically what happens is that all layer are the same. There could be some short-term
#    difference in kLight between the surface and at depth, but in general vertical mixing
#    will make the time-average kLight approximately uniform. That makes growth rate uniform,
#    and there's really no difference between looking at the surface layer or the vertical mean.
#  - what happens if there is extreme swimming, all P is in the top layer, but has access
#    to N from the entire water column?
#    In this case the biomass-weighted average growth rate will be the growth rate of the
#    tracer in the top layer. The growth rate in all of the layer will be higher than in the
#    mixed case, and the growth rate in the top layer biased even higher. The mean P concentration
#    will be much smaller than the concentration in the surface, and relative to the N initial
#    condition it will have more nutrients.
#  - other scenarios? There could be some errors in the in-between cases, where exponential
#    growth at the surface outpaces that at depth, and the biomass-weighted growth rate
#    essentially varies with time.
# 
# This is unlikely to come out exact - we're looking at a sum of exponentials with different exponents. But
# given that there will be periodic vertical mixing, maybe it's not such a bad approximation.
# 
# The key points to check/implement are:
# 
#  1. layer='mean' should account for the differing layer thicknesses
#  2. kLight calculation should be a biomass-weighted average of the per-layer growth rates.
# 
# Turns out step (1) was already there. For step (2), the existing code does a biomass-weighted
# average of kRadA and kRadB. It's not quite the same as calculating kLight per layer and then 
# biomass-weighted avearge of the per-layer values. But I think it's not that far off. Better
# to forge ahead with the optimization
# 
# Come back to taking a closer look at the vertical integration, work through the calculations
# by hand.
# 
# ```
#             constant_params['kLight']=bloom_common.calc_kLight(Isat=constant_params['Isat'],
#                                                                kRadA=row.kRadA,IsatA=row.IsatA,
#                                                                kRadB=row.kRadB,IsatB=row.IsatB,
#                                                                conc=row.conc,age_d=row.age_d)
# ```
