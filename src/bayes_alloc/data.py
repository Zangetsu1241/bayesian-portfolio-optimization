"""Synthetic correlated assets and strictly validated price CSV."""
import numpy as np
import pandas as pd

def synthetic_prices(days=850,seed=42):
    if days<120:raise ValueError('days must be >=120')
    rng=np.random.default_rng(seed)
    assets=['equity','bonds','gold','commodities']
    vols=np.array([[.012,.004,.009,.014],[.024,.010,.015,.028]])
    means=np.array([[.00045,.0001,.00018,.00025],[-.00065,.0003,.0004,-.0008]])
    corr=np.array([[1,-.1,.2,.55],[-.1,1,.05,-.1],[.2,.05,1,.15],[.55,-.1,.15,1.]])
    state=0
    r=np.empty((days,4))
    for t in range(days):
        if rng.random()<.015:state=1-state
        r[t]=rng.multivariate_normal(means[state],np.outer(vols[state],vols[state])*corr)
    return pd.DataFrame(100*np.exp(np.cumsum(r,axis=0)),columns=assets,index=pd.bdate_range('2018-01-01',periods=days))

def load_prices(path):
    frame=pd.read_csv(path)
    if 'date' not in frame or len(frame.columns)<3:raise ValueError('Need date and >=2 asset price columns')
    dates=pd.to_datetime(frame.pop('date'),utc=True,errors='raise')
    numeric=frame.apply(pd.to_numeric,errors='raise')
    x=numeric.to_numpy(float)
    if (dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or
        np.any(~np.isfinite(x)) or np.any(x<=0) or len(frame)<120):
        raise ValueError('Need >=120 rows, strictly ordered unique dates and positive finite prices')
    numeric.index=dates
    return numeric
