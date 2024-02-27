import os, glob, shutil
import xml.etree.ElementTree as ET
import datetime
import six
import xarray as xr
import pandas as pd

from stompy.spatial import field
from stompy import utils, memoize
import stompy.model.delft.dflow_model as dfm
import stompy.model.delft.waq_scenario as dwaq
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import logging as log
from scipy.interpolate import griddata
from stompy.spatial import proj_utils
import numpy as np

chl_data_dir="rs_chl/04b_log10"

# supposedly gdal understands dimap, and these are supposedly
# dimap scenes. Not working. But we can load the pieces and 
# put back together.


def acquisition_period(rec):
    xml_fn=rec['xml']
    tree = ET.parse(xml_fn)
    root = tree.getroot()
    def parse_date(s):
        return np.datetime64( datetime.datetime.strptime(s.split('.')[0],"%d-%b-%Y %H:%M:%S"))

    for tag in root.iter("PRODUCT_SCENE_RASTER_START_TIME"):
        scene_start_time = parse_date(tag.text)
    for tag in root.iter("PRODUCT_SCENE_RASTER_STOP_TIME"):
        scene_stop_time = parse_date(tag.text)
    return pd.Series(dict(acquisition_start=scene_start_time,
                          acquisition_stop =scene_stop_time))

def load_chl_scenes():
    # These are xml formatted, with tons of metadata
    scenes=glob.glob(os.path.join(chl_data_dir,"*.dim"))
    scenes.sort()

    xml_df=pd.DataFrame()
    xml_df['xml']=scenes

    times=xml_df.apply(acquisition_period,axis=1)
    scene_df=pd.concat([xml_df,times],axis=1)
    return scene_df



def load_scene(dim_path,fill_coordinates=True):
    scene_dir=dim_path.replace('.dim','.data')
    #os.path.join(chl_data_dir,"20220821_BandMath_BandMath.data")

    layers=glob.glob(os.path.join(scene_dir,"*.img"))
    layers.sort() # chl, lat, long

    layer_flds=[field.GdalGrid(layer) for layer in layers]
    # Looks like missing data is 0.0? and remaining data is log10 transformed
    layer_flds[0].F = np.where(layer_flds[0].F==0.0, np.nan, 10**layer_flds[0].F)
    # Also latitude,longitude are stored as integers
    layer_flds[1].F = 1e-6 * layer_flds[1].F
    layer_flds[2].F = 1e-6 * layer_flds[2].F
    
    xyz=np.stack( [layer_flds[2].F, layer_flds[1].F, layer_flds[0].F]).transpose([1,2,0])

    if fill_coordinates:
        ll=xyz[:,:,:2]
        valid=(np.isfinite(ll)) & (ll>-185) & (ll<365)
        valid=np.all(valid,axis=2)
        utils.fill_curvilinear(xyz[:,:,:2],xy_valid=valid)
    return xyz

@memoize.memoize(lru=10)
def load_scene_utm(dim_path,clip=None,ravel=True):
    """
    project to utm with optional ravel and clip
    """
    xym=load_scene(dim_path)
    if ravel:
        xym=xym.reshape(-1,3)
    utm_xy = proj_utils.mapper("WGS84","EPSG:26910")(xym[...,:2])
    m=xym[...,2]
    if clip is not None:
        m=m.clip(0,clip)
    if ravel:
        return np.c_[utm_xy[:,0],utm_xy[:,1],m]
    else:
        xym=xym.copy()
        xym[...,:2] = utm_xy
        xym[...,2] = m
        return xym

def configure_dfm_t140737():
    DELFT_SRC="/opt/software/delft/dfm/t140737"
    DELFT_SHARE=os.path.join(DELFT_SRC,"share","delft3d")
    DELFT_LIB=os.path.join(DELFT_SRC,"lib")

    os.environ['DELFT_SRC']=DELFT_SRC
    os.environ['DELFT_SHARE']=DELFT_SHARE

    # While mucking around with this just clobber whatever was in LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH']=f"{DELFT_LIB}:/home/rusty/.conda/envs/dfm_t140737/lib"

def configure_dfm_2023_01():
    DELFT_SRC="/opt/software/delft/dfm/2023.01"
    DELFT_SHARE=os.path.join(DELFT_SRC,"share","delft3d")
    DELFT_LIB=os.path.join(DELFT_SRC,"lib")

    os.environ['DELFT_SRC']=DELFT_SRC
    os.environ['DELFT_SHARE']=DELFT_SHARE

    # While mucking around with this just clobber whatever was in LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH']=DELFT_LIB
 
def plot_xyz(xyz,ax=None,**kw):
    # expand pixel center ll to pixel corners
    x_corners,y_corners=utils.center_to_edge_2d(xyz[:,:,0],xyz[:,:,1])
    if ax is None:
        ax=plt.gca()
    return ax.pcolormesh( x_corners, y_corners, xyz[...,2], **kw)
