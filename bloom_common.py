import os, glob, shutil
import xml.etree.ElementTree as ET
import datetime
import six
import xarray as xr
import pandas as pd
from shapely import geometry, wkt

from stompy.spatial import field
from stompy import utils, memoize
import stompy.model.delft.dflow_model as dfm
import stompy.model.delft.waq_scenario as dwaq
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import logging as log
from scipy.interpolate import griddata
from scipy import ndimage
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

def load_chl_scenes(chl_data_dir=chl_data_dir):
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

def configure_dfm_t141798():
    DELFT_SRC="/opt/anaconda3/envs/dfm_t141798"
    #DELFT_SRC="/opt/anaconda3/envs/dfm_t141798opt" # fort-collins doesn't have the base compile, but does have opt
    DELFT_SHARE=os.path.join(DELFT_SRC,"share","delft3d")
    DELFT_LIB=os.path.join(DELFT_SRC,"lib")

    # So we can do custom processes
    os.environ['PROC_TABLE_SRC_DIR']=os.path.join(DELFT_SRC,"build/dfm/src/src/engines_gpl/waq/default/csvFiles")
    #os.environ['PROC_TABLE_SRC_DIR']="/richmondvol1/rusty/csvFiles-141798" # for fort-collins and others
    os.environ['DELFT_SRC']=DELFT_SRC
    os.environ['DELFT_SHARE']=DELFT_SHARE
    os.environ['LD_LIBRARY_PATH']=DELFT_LIB


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

ssfb_poly=wkt.loads("""Polygon ((552701.88427324150688946 4182777.26593696838244796, 557664.74622329045087099 4184715.14536508312448859, 558704.58396520547103137 4181690.16284314822405577, 563052.9963404864538461 4179279.62989598186686635, 564470.95689764339476824 4179232.36454407637938857, 564612.75295335904229432 4177058.15835643606260419, 565888.91745480021927506 4175025.74822451127693057, 569244.75744007143657655 4172662.48062924947589636, 570709.9833491335157305 4173324.19555592304095626, 574254.88474202563520521 4169873.82486684108152986, 575483.78389156155753881 4162122.30715438397601247, 576759.94839300273451954 4157490.3026676713488996, 578934.15458064316771924 4153898.13592287385836244, 574396.68079774128273129 4150022.37706664530560374, 570520.92194151261355728 4154323.52409002091735601, 565794.38675098970998079 4157726.62942719738930464, 561540.50507951923646033 4158908.2632248280569911, 556152.2549623231170699 4160420.75448579574003816, 553741.72201515652704984 4163209.41024820413440466, 553316.33384800946805626 4168314.06825396884232759, 552701.88427324150688946 4182777.26593696838244796))""")


# Regions (copy paste from QGIS vector layer)
oakland_poly=wkt.loads("""Polygon ((560434.67213132162578404 4183884.9798020776361227, 
567612.90416294464375824 4183342.8737371894530952, 571557.19311782089062035 
4177809.65321281319484115, 570472.98098804452456534 4175697.30889100721105933, 
569052.2892317856894806 4173491.49800628982484341, 567612.90416294464375824 
4172893.3120036544278264, 565014.53371399780735373 4172930.69862881954759359, 
562920.88270477438345551 4174201.84388441918417811, 562528.32314054504968226 
4175510.37576518394052982, 562976.96264252148102969 4181174.44947763625532389,
560042.11256709217559546 4182651.2211716421879828, 560434.67213132162578404 
4183884.9798020776361227))
""")

eastshore_poly=wkt.loads("""Polygon ((568776.56287119572516531 4174108.37732150498777628,
570122.4813771250192076 4175678.61557842278853059, 573487.2776419484289363
4172538.13906458765268326, 575842.63502732466440648 4169790.22211498161777854, 
575898.71496507176198065 4167042.30516537604853511, 577188.55353325395844877 
4160312.71263572946190834, 577917.59272396576125175 4160144.47282248828560114, 
582460.06768147717230022 4158125.59506359416991472, 580777.66954906552564353 
4156667.51668217079713941, 577188.55353325395844877 4157172.23612189432606101, 
576627.75415578344836831 4156555.3568066768348217, 575618.31527633650694042 
4157789.11543711181730032, 574664.95633463654667139 4160200.55276023549959064, 
574160.23689491301774979 4163060.62958533503115177, 573767.67733068368397653 
4165920.7064104350283742, 572814.3183889837237075 4169509.82242624647915363, 
570683.28075459564570338 4171416.54030964663252234, 568776.56287119572516531 
4174108.37732150498777628))
""")


def set_minimal_map_output(model):
    # Trim the map output down to keep files smaller
    model.mdu['output','Wrimap_waterlevel_s0']=0 # no need for last step's water level
    model.mdu['output','Wrimap_velocity_component_u0']=0 #        # Write velocity component for previous time step to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_velocity_component_u1']=0 #        # Write velocity component to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_velocity_vector']=0 #              # Write cell-center velocity vectors to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_upward_velocity_component']=0 #    # Write upward velocity component on cell interfaces (1: yes, 0: no)
    model.mdu['output','Wrimap_density_rho']=0 #                  # Write flow density to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_horizontal_viscosity_viu']=0 #     # Write horizontal viscosity to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_horizontal_diffusivity_diu']=0 #   # Write horizontal diffusivity to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_flow_flux_q1']=0 #                 # Write flow flux to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_spiral_flow']=0 #                  # Write spiral flow to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_chezy']=0 #                        # Write the chezy roughness to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_turbulence']=0 #                   # Write vicww, k and eps to map file (1: yes, 0: no)
    model.mdu['output','Wrimap_wind']=0 #                         # Write wind velocities to map file (1: yes, 0: no)

    model.mdu['output','Wrimap_velocity_component_u1']=0
    model.mdu['output','Wrimap_velocity_vector']=0
    model.mdu['output','Wrimap_velocity_magnitude']=0
    model.mdu['output','Wrimap_upward_velocity_component']=0
    model.mdu['output','Wrimap_heat_fluxes']=0


class SFBRestartable(dfm.DFlowModel):
    """
    Add special sauce to symlink relevant forcing files that are
    not standard for dflow_model.
    This includes bc_files, src_files, meteo_coarse.grd
    """
    restart_copy_names=["source_files"] # symlink everything

    def copy_files_for_restart(self):
        super().copy_files_for_restart()
        prev_model=self.restart_from
        # Subdirectories are not automatically copied over, as well as meteo_coarse.grd
        for sub in ['bc_files','source_files','meteo_coarse.grd']:
            src=os.path.join(prev_model.run_dir,sub)
            if not os.path.exists(src):
                self.log.warning(f"Expected to symlink {src} but it wasn't there.")
                continue
            dst=os.path.join(self.run_dir,sub)
            if os.path.exists(dst):
                self.log.warning(f"Expected to make {dst} a copy or symlink but it already exists")
                continue

            if sub in self.restart_copy_names:
                self.log.info(f"Copy {src} => {dst}")
                shutil.copytree(src,dst)
            else:
                # TODO: chase down symlinks to get back to the real file
                # otherwise this will break if intermediate runs are removed.
                # Make the symlinks relative in all of this moves, is on a different machine, etc.
                src_rel=os.path.relpath(src,start=self.run_dir)
                self.log.info(f"Symlink {dst} => {src_rel}")
                os.symlink(src_rel,dst)
                
def CART(name="CART1",
         zero_order="ZAge1",
         partial="PartAge1",
         conc="Age1Conc",
         age_conc="Age1AConc",
         flux="dAge1",
         conc_decay=0.0):
    n_stoich=2
    if conc_decay!=0.0:
        #conc_stoich=f"ALKA1       {flux:10}    -8.71400"
        n_stoich+=1
    process=f"""
{name}                    Reuse nitrification as age                         
NITRIF    ; module name. 
123       ; TRswitch
        18; # input items for segments
{zero_order:10}      0.00000     x zeroth-order nitrification flux          (gN/m3/d)
{partial   :10}      1.00000     x set this to get partial age
RcNit20        0.100000     x ignored (b/c SWVnNit=0)
TcNit           1.00000     x ignored
OXY             10.0000     x ignored
KsAmNit        0.500000     x ignored
KsOxNit         1.00000     x ignored
Temp            15.0000     x ignored
CTNit           3.00000     x ignored
Rc0NitOx        0.00000     x ignored
COXNIT          1.00000     x ignored
Poros           1.00000     x volumetric porosity                            (-)
SWVnNit         0.00000     x switch for old (0), new (1), TEWOR (2) version (-)
{conc    :10}     0.100000     x concentration tracer
OOXNIT          5.00000     x ignored
CFLNIT          0.00000     x ignored
CurvNit         0.00000     x ignored
DELT           -999.000     x timestep for processes                         (d)
         0; # input items for exchanges
         1; # output items for segments
O2FuncNT1                   x oxygen function for nitrification              (-)
         0; # output items for exchanges
         1; # fluxes
{flux:10}                  x nitrification flux                       (gN/m3/d)
         {n_stoich}; # stoichiometry lines. Could probably drop most of these.
NH4         {flux:10}    -1.00000
{age_conc:10}  {flux:10}     1.00000
"""
    if conc_decay!=0.0:
        process+=f"{conc:10}  {flux:10}    {-conc_decay:.5f}\n"
    process+="""         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
    return process



# Post-processing:
@memoize.memoize(lru=5)
def load_model(run_dir):
    model = dfm.DFlowModel.load(run_dir)

    grid=model.grid
    M=grid.smooth_matrix()
    @utils.add_to(model)
    def fill(self,values,iterations=50):
        valid=np.isfinite(values)
        data=np.where(valid,values,0.0)
        weight=np.where(valid,1,0.0)
        for _ in range(iterations):
            data=M.dot(data)
            weight=M.dot(weight)
            data[valid]=values[valid]
            weight[valid]=1.0
        result=np.full(len(values),np.nan)
        valid=weight>1e-4
        result[valid]=data[valid]/weight[valid]
        return result

    return model

def ratio(a,b,b_min=1e-8):
    a_b = a/b.clip(b_min)
    # avoid using np.where, since it loses xarray dimensions.
    a_b[b<b_min] = np.nan
    return a_b



# rs_chl_dir="/richmondvol1/lawrences/outputs_2022/07_reproject_3"
# 2024-09-11: 07_reproject_3 no longer exists.
# Had been using this:
# rs_chl_dir="/richmondvol1/lawrences/outputs_2022/07_reproject_re10_sfei"
# 2024-10-11: Dave suggests this is the best to use:
rs_chl_dir="/richmondvol1/lawrences/outputs_2022/07_reproject_re10_raphe"

def chl_from_RS(t,grid_poly):
    patt = os.path.join(rs_chl_dir, f"{utils.strftime(t,'%Y%m%d')}_S3*_OL_NT_CHL_LOG10.img")
    hits=glob.glob(patt)
    if not hits:
        return None
    #scene="20220807_S3A_OL_NT_CHL_LOG10.img" # decent, though getting late.
    #rs_chl_fn=os.path.join(rs_chl_dir,scene)
    rs_chl_fn=hits[0]
    rs_chl_log10 = field.GdalGrid(rs_chl_fn)
    
    # clip and remove specks.    
    valid = np.isfinite(rs_chl_log10.F)
    valid = valid & rs_chl_log10.polygon_mask(grid_poly) 
    rs_chl_log10.F[~valid]=np.nan
    rs_chl_log10.F = ndimage.median_filter(rs_chl_log10.F,size=3)
    # rs_chl_log10.F[ rs_chl_log10.F<1.0 ] = 0.0
        
    chl_IC = rs_chl_log10.copy()
    chl_IC.F = 10**chl_IC.F    
        
    return chl_IC
