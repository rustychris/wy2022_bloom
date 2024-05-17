"""
Various function used in multiple notebooks
"""
import numpy as np
import xarray as xr

# See plot_transport_and_swimming.ipynb for notes on derivation.

def k_indef(z,Kd,I0,Isat):
    """
    solution of indefinite integral
      int I(z)/(Isat+I(z)) dz
    where I(z) = I0 exp(-Kd*z).

    z is taken as distance from the surface (depth).
    """
    
    # Avoid divzero (creates a lower bound of about 1e-15 on light limitation)
    #I0=np.clip(I0,1e-12,2000)
    if isinstance(z,np.ndarray):
        z=z.astype(np.float64)
    # logaddexp does a transformation behind the scenes that makes it numerically stable.
    return  z - 1/Kd * np.logaddexp( Kd*z+np.log(Isat/I0), np.log(1))
    # numerically unstable:
    # return z - 1/Kd * np.log(Isat/I0*np.exp(Kd*z)+1)

def kLight_m(Kd,h_min,h_max,I0,Isat=1000/25.):
    """
    Instantaneous light limitation function, depth-integrated.
    K_d: 1/m light attenuation
    h_min: depth to top of segment (positive-down)
    h_max: depth to bottom of segment (positive-down)
    I0: surface irradiance
    Isat: half-saturation for irradiance
    
    return value has units of meters (consistent with equations above.
    it is meant to be normalized by depth outside the summation.)
    """
    return k_indef(h_max,Kd,I0,Isat) - k_indef(h_min,Kd,I0,Isat)

def kLight(Kd,h_min,h_max,*a,**kw):
    """
    Depth-averaged version of above
    """
    return kLight_m(Kd,h_min,h_max,*a,**kw) / (h_max - h_min)

def sigma_median(C):
    # C is assumed to be cell-constant
    # assume vertical coordinate is the last dimension
    sigma=np.linspace(0,1,1+C.shape[-1])
    accum = np.cumsum(C,axis=-1)
    accum = np.concatenate( [ 0*accum[...,:1], accum], axis=-1)
    if C.ndim==1:
        return np.interp(0.5, accum/accum[-1], sigma)
    else:
        results = np.zeros( C.shape[:-1], np.float64)
        for idx in np.ndindex(*results.shape):
            results[idx] = np.interp( 0.5, accum[idx][:]/accum[idx][-1], sigma)
        return results # np.interp(0.5, accum/accum[-1], sigma)


def resample(ds,dt):
    #delegating to pandas is surprisingly slow and ornery.
    #noaa_alameda.resample({'time':'360s'}).mean()

    t_new = np.arange(ds.time.values.min(), 
                      ds.time.values.max(),
                      dt)
    ds_resample = xr.Dataset()
    ds_resample['time']=('time',),t_new

    freq=f"{dt/np.timedelta64(1,'s'):.0f}s"
    for col in ds.data_vars:
        #print(col)
        if 'time' not in ds[col].dims:
            ds_resample[col] = ds[col].copy() # don't share data
            continue
        # how bad is it to batch off to pandas?
        resampled=ds[col].to_dataframe().resample(freq).mean()
        ds_resample[col]= ds[col].dims, resampled[col].values
    return ds_resample

