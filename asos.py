# Sample URL:
# https://www.ncei.noaa.gov/data/automated-surface-observing-system-five-minute/access/2022/08/asos-5min-KOAK-202208.dat

import os
import requests
import time
from stompy import utils
from io import StringIO
import numpy as np
import re
import datetime
import pandas as pd


def fetch_asos_txt(station,year,month,cache_dir="cache"):
    basename = f"asos-5min-{station}-{year:04d}{month:02d}.dat"
    cache_path = os.path.join(cache_dir,basename)
    
    if not os.path.exists(cache_path):
        url=f"https://www.ncei.noaa.gov/data/automated-surface-observing-system-five-minute/access/{year:04d}/{month:02d}/{basename}"
        utils.download_url(url, cache_path, on_abort='remove')
        time.sleep(0.5)

    with open(cache_path,'rt') as fp:
        return fp.read()

def fetch_asos_df(station,year,month):
    raw = fetch_asos_txt(station,year,month)
    return parse_asos(raw)
    
def parse_asos(raw):
    records=[]
    conditions={'CLR':0.025,  # 0-5%
                'FEW':0.15,   # 5-25%
                'SCT':0.375,  # 25-50%
                'BKN':0.685,  # 50-87%
                'OVC':0.94    # 87-100%
               }
    for line in StringIO(raw):
        cloudiness = -1
        for sky_phrase in re.findall(r'CLR|FEW|SCT|BKN|OVC', line):
            cloudiness = max(cloudiness, conditions[sky_phrase])
        if cloudiness<0: continue            
        # parse a timestamp
        # ASOS data are in LST. Add 8h to get UTC
        t=datetime.datetime.strptime(line[13:25],"%Y%m%d%H%M") + datetime.timedelta(hours=8)
        records.append( dict(time=t,cloudiness=cloudiness) )
    
    return pd.DataFrame(records)

def fetch_asos(station,start,stop):
    dt_start=utils.to_datetime(start)
    dt_stop =utils.to_datetime(stop)

    t=datetime.datetime(year=dt_start.year, month=dt_start.month, day=1, hour=0, minute=0)

    dfs=[]
    while t < dt_stop:
        for retry in range(1):
            try:
                dfs.append(fetch_asos_df(station,t.year,t.month))
                break
            except requests.HTTPError:
                print("HTTPError") # mostly 404
        month0 = t.month - 1 + 1
        t = t.replace(year=t.year+month0//12, month=1 + (month0%12))

    return pd.concat(dfs)
