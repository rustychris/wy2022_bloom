import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr
import pandas as pd
import bloom_common

def solveNP(group,
            thresh=1e-5,
            # Parameters:
            c0 = 1,
            tidx = 20, 
            Isat=20., 
            N0=35, # initial DIN
            Nsat=0.5,
            kprod=1.0, # per day assuming 24h light. so 0.5 d^-1 in typical lit.
            kmort=0.1,
            background=0.0,
            alpha=0.15, # stoichiometry, uM DIN per ug/l chl production.
            light_lim='mean_lim',
            layer=15,
            presel=None, # bitmask over cells, True to compute. False entries get nan.
            P_nodata=0.0,
            N_nodata=None, # N0
            label=None, # ignored.
           ):
    tracers=group.extract_tracers(tidx=tidx, layer=layer, Isat=Isat, thresh=thresh, light_lim=light_lim)
    kLight = tracers['kLight']
    conc = c0*tracers['conc']
    age_d = tracers['age_d']
    t = tracers['t']

    if N_nodata is None:
        N_nodata=N0
        
    if age_d>0.0:
        sel = np.isfinite(kLight * conc * age_d)
    else:
        sel = np.isfinite(conc)

    if presel is not None:
        sel = presel & sel

    kLight = kLight[sel]

    # ODE integration
    # State vector is [ P[0].... P[i], N[0], ... , N[i] ]
    def diff(t,state):
        P,N=state.reshape([2,-1])
        N=N.clip(0) # no evidence this is really necessary
        P=P.clip(0) # likewise.
        kDIN=N/(N+Nsat)
        dgrossP = kprod*kLight*kDIN*P
        dnetP = -kmort*P + dgrossP
        dN = -alpha*dgrossP
        mu_net = kprod*kLight*kDIN - kmort
        return np.r_[dnetP,dN]
    IC=np.r_[conc[sel], N0*np.ones_like(conc)[sel]]

    # print("IC shape: ", IC.shape) # 2*Ncells

    if age_d>0:
        # odeint was buggy.
        #result = odeint(diff, IC, [0,age_d], hmax=0.1)
        bunch = solve_ivp(diff, y0=IC, t_span=[0,age_d])
        result=bunch.y[:,-1]
    else:
        result = IC
    Psel,Nsel = result.reshape([2,-1])
    # expand
    # assume no biomass, no depletion of N when conc too small.
    P = np.full(conc.shape,P_nodata)
    N = np.full(conc.shape,N_nodata)
        
    P[sel]=Psel + background # or add to P?
    N[sel]=Nsel
    
    ds=xr.Dataset()
    ds['N'] = ('cell',), N
    ds['P'] = ('cell',), P
    ds['time'] = (),t

    ds['c0'] = (),c0
    ds['Isat'] = (),Isat
    ds['N0'] = (),N0
    ds['Nsat'] = (),Nsat
    ds['kprod'] = (), kprod
    ds['kmort'] = (), kmort
    ds['alpha'] = (), alpha
    ds['layer'] = (), layer
    
    return ds


# This version gets applied to a DataFrame of inputs, returns matching
# dataframe with output. Also includes code for exploring other source/sink
# terms.
# df_input is expected to come in with conc, age_d, kLight
def solveNPZ_df(df_input,
                npz_diff,
                thresh=1e-5,
                # Parameters:
                c0 = 1,
                tidx = 20, 
                Isat=20., 
                N0=35, # initial DIN
                Nsat=0.5,
                kprod=1.0, # per day assuming 24h light. so 0.5 d^-1 in typical lit.
                kmort=0.1,
                background=0.0,
                alpha=0.15, # stoichiometry, uM DIN per ug/l chl production.
                light_lim='mean_lim',
                layer=15,
                P_nodata=0.0,
                N_nodata=None, # N0
                label=None, # ignored.
                Z0=0.0,
                mu_Z=0.1, # per-day
                ):
    if N_nodata is None:
        N_nodata=N0

    # ODE integration
    # State vector is [N, P, Z]
    # example diff
    # def diff(t, state, kLight):
    #     P,N,Z=np.clip(state,0,None)
    #     
    #     kDIN=N/(N+Nsat)
    #     dgrossP = kprod*kLight*kDIN*P
    #     dnetP = -kmort*P + dgrossP
    #     dN = -alpha*dgrossP
    #     mu_net = kprod*kLight*kDIN - kmort
    #     return np.r_[dnetP,dN,0.0]
    
    common=dict(c0=c0, Isat=Isat, N0=N0, Z0=Z0, Nsat=Nsat, kprod=kprod, kmort=kmort,
                alpha=alpha, layer=layer)
    recs=[]

    for i,row in enumerate(df_input.itertuples()):
        IC=np.r_[N0, c0*row.conc, Z0]
        age_d=row.age_d
        kLight=bloom_common.calc_kLight(Isat=Isat,kRadA=row.kRadA,IsatA=row.IsatA,
                                        kRadB=row.kRadB,IsatB=row.IsatB,
                                        conc=row.conc,age_d=row.age_d)
        if np.isnan(kLight):
            result = np.nan*IC
        elif age_d>0:
            bunch = solve_ivp(npz_diff, y0=IC, t_span=[0,age_d], args=(kLight,))
            result=bunch.y[:,-1]
        else:
            result = IC
        recs.append(dict(N=result[0],P=result[1]+background,Z=result[2],
                         time=row.time, kLight=kLight)
                    | common)
    df_output=pd.DataFrame(recs)
    
    return df_output
