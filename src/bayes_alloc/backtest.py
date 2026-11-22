"""Rolling, historical-only allocation and portfolio performance accounting."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .models import (TRADING_DAYS,covariance_estimate,black_litterman,
    bayesian_mean,optimal_weights,minimum_variance,default_absolute_views)

@dataclass(frozen=True)
class Config:
    lookback: int = 126
    rebalance_every: int = 21
    cost_bps: float = 5.
    risk_aversion: float = 3.
    tau: float = .05
    shrinkage: float = .2
    max_weight: float = .8
    view_lookback: int = 21
    view_uncertainty: float = 2.
    prior_strength: float = 20.

def walk_forward(prices, cfg=Config()):
    if not isinstance(prices,pd.DataFrame) or prices.shape[1]<2:
        raise ValueError('Expected a 2+-asset DataFrame')
    arr=prices.to_numpy(float)
    if np.any(~np.isfinite(arr)) or np.any(arr<=0) or not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError('Positive finite prices with ordered unique dates required')
    if (cfg.lookback<30 or cfg.rebalance_every<1 or cfg.view_lookback<3 or
        cfg.view_lookback>cfg.lookback or cfg.cost_bps<0 or cfg.risk_aversion<=0 or
        cfg.tau<=0 or not 0<=cfg.shrinkage<=1 or not 1/prices.shape[1]<=cfg.max_weight<=1 or
        cfg.view_uncertainty<=0 or cfg.prior_strength<=0):
        raise ValueError('Invalid configuration')
    rets=prices.pct_change().dropna()
    if len(rets)<=cfg.lookback:raise ValueError('Insufficient prices for lookback')
    n=prices.shape[1]
    names=['equal_weight','minimum_variance','sample_mean','bayesian_mean','black_litterman_prior','black_litterman_views']
    holdings={name:np.ones(n)/n for name in names}
    records=[];weights=[]
    # t is close-to-close realized from t-1 to t. Uses only observations <=t.
    for t in range(cfg.lookback-1,len(rets)-1):
        next_returns=rets.iloc[t+1].to_numpy(float)
        rebalance=(t-(cfg.lookback-1))%cfg.rebalance_every==0
        if rebalance:
            history=rets.iloc[t-cfg.lookback+1:t+1].to_numpy(float)
            cov=covariance_estimate(history,cfg.shrinkage)
            market=np.ones(n)/n
            pi,bl_prior_cov=black_litterman(cov,market,cfg.risk_aversion,cfg.tau)
            P,Q,omega=default_absolute_views(history,cov,cfg.tau,cfg.view_uncertainty,cfg.view_lookback)
            bl_mean,bl_cov=black_litterman(cov,market,cfg.risk_aversion,cfg.tau,P,Q,omega)
            bayes_mu,bayes_cov=bayesian_mean(history,cov,prior_mean=pi,prior_strength=cfg.prior_strength)
            targets={
                'equal_weight':market,
                'minimum_variance':minimum_variance(cov,cfg.max_weight),
                'sample_mean':optimal_weights(history.mean(axis=0)*TRADING_DAYS,cov,cfg.risk_aversion,cfg.max_weight),
                'bayesian_mean':optimal_weights(bayes_mu,cov+bayes_cov,cfg.risk_aversion,cfg.max_weight),
                'black_litterman_prior':optimal_weights(pi,cov+bl_prior_cov,cfg.risk_aversion,cfg.max_weight),
                'black_litterman_views':optimal_weights(bl_mean,cov+bl_cov,cfg.risk_aversion,cfg.max_weight)
            }
        for name in names:
            w=holdings[name]
            turnover=0.
            if rebalance:
                target=targets[name]
                turnover=float(np.abs(target-w).sum())
                w=target
            gross=float(w@next_returns)
            net=gross-cfg.cost_bps/10000*turnover
            if net<=-1 or gross<=-1:raise ValueError('Bankrupt portfolio')
            records.append({'date':rets.index[t+1],'signal_date':rets.index[t],
                            'strategy':name,'gross_return':gross,'net_return':net,
                            'turnover':turnover,'rebalance':rebalance})
            weights.append({'date':rets.index[t+1],'strategy':name,**dict(zip(prices.columns,w))})
            holdings[name]=w*(1+next_returns)/(1+gross)
    return pd.DataFrame(records).set_index('date'),pd.DataFrame(weights).set_index('date')

def metrics(net_returns,turnover=None):
    x=np.asarray(net_returns,float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all() or np.any(x<=-1):
        raise ValueError('Expected finite nonbankrupt daily simple returns')
    equity=np.cumprod(1+x)
    peak=np.maximum.accumulate(np.r_[1.,equity])[1:]
    sd=float(np.std(x,ddof=1)) if len(x)>1 else 0.
    out={'total_return':float(equity[-1]-1),
         'annualized_return':float(equity[-1]**(TRADING_DAYS/len(x))-1),
         'annualized_volatility':sd*np.sqrt(TRADING_DAYS),
         'sharpe':float(np.mean(x)/sd*np.sqrt(TRADING_DAYS)) if sd>0 else 0.,
         'max_drawdown':float(np.min(equity/peak-1))}
    if turnover is not None:out['turnover']=float(np.asarray(turnover).sum())
    return out
