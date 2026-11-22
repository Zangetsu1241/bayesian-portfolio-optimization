import numpy as np
import pytest
from bayes_alloc.models import (black_litterman,bayesian_mean,covariance_estimate,
    optimal_weights,minimum_variance,default_absolute_views)
from bayes_alloc.data import load_prices,synthetic_prices
from bayes_alloc.backtest import Config,walk_forward,metrics
from bayes_alloc.cli import main

COV=np.array([[.04,.004,.005],[.004,.02,.002],[.005,.002,.03]])
MARKET=np.array([.4,.35,.25])

def test_bl_without_views_returns_prior():
    mu,unc=black_litterman(COV,MARKET,risk_aversion=3.,tau=.1)
    assert np.allclose(mu,3*COV@MARKET)
    assert np.allclose(unc,.1*COV)

def test_bl_high_confidence_view_moves_mean():
    P=np.array([[1.,0.,0.]])
    old,_=black_litterman(COV,MARKET)
    new,post=black_litterman(COV,MARKET,P=P,Q=np.array([.20]),omega=np.array([[1e-8]]))
    assert abs(new[0]-.20)<1e-4
    assert abs(new[0]-.20)<abs(old[0]-.20)
    assert np.linalg.eigvalsh(post).min()>0

def test_bl_uncertain_views_approach_prior():
    prior,_=black_litterman(COV,MARKET)
    posterior,_=black_litterman(COV,MARKET,P=np.eye(3),Q=np.full(3,.2),omega=np.eye(3)*1e9)
    assert np.allclose(posterior,prior,atol=1e-8)

def test_bl_view_order_invariance():
    P=np.array([[1.,-1.,0.],[0.,1.,-1.]])
    Q=np.array([.02,.015]);o=np.diag([.001,.002])
    a,ca=black_litterman(COV,MARKET,P=P,Q=Q,omega=o)
    b,cb=black_litterman(COV,MARKET,P=P[::-1],Q=Q[::-1],omega=o[::-1,::-1])
    assert np.allclose(a,b)
    assert np.allclose(ca,cb)

def test_bayesian_mean_conjugacy():
    x=np.ones((20,3))*.01
    mu,u=bayesian_mean(x,COV,prior_mean=np.zeros(3),prior_strength=20)
    assert np.allclose(mu,np.ones(3)*.01*252/2)
    assert np.allclose(u,COV*252/40)

def test_bayesian_more_data_less_uncertainty():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(300,3))*.01
    _,a=bayesian_mean(x[:40],COV)
    _,b=bayesian_mean(x,COV)
    assert np.trace(b)<np.trace(a)

def test_optimization_constraints_and_utility():
    mu=np.array([.1,.04,.06])
    w=optimal_weights(mu,COV,max_weight=.6)
    assert np.isclose(w.sum(),1) and np.all(w>=-1e-12) and np.max(w)<=.600001
    eq=np.ones(3)/3
    def objective(v):return mu@v-1.5*v@COV@v
    assert objective(w)>=objective(eq)-1e-9

def test_minimum_variance_risk_reduction():
    w=minimum_variance(COV)
    assert w@COV@w <= np.ones(3)@COV@np.ones(3)/9+1e-8

def test_covariance_regularization():
    rng=np.random.default_rng(5)
    a=rng.normal(size=(80,1))
    x=np.repeat(a,3,axis=1)
    cov=covariance_estimate(x)
    assert np.linalg.eigvalsh(cov).min()>0

def test_views_use_training_suffix_only():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(200,3))
    c=covariance_estimate(x[:100])
    p,q,o=default_absolute_views(x[:100],c)
    changed=x.copy();changed[101:]+=500
    p2,q2,o2=default_absolute_views(changed[:100],c)
    assert np.array_equal(q,q2)
    assert np.array_equal(o,o2)

def test_backtest_accounting_and_alignment():
    p=synthetic_prices(235)
    ret,weights=walk_forward(p,Config(lookback=65,rebalance_every=15))
    assert len(ret)==len(weights)
    assert (ret.signal_date<ret.index).all()
    assert np.allclose(ret.net_return,ret.gross_return-ret.turnover*.0005)
    assert np.allclose(weights.iloc[:,1:].sum(axis=1),1,atol=1e-8)
    assert ret.strategy.nunique()==6

def test_prefix_invariance():
    prices=synthetic_prices(240)
    cfg=Config(lookback=65,rebalance_every=15)
    shortened=prices.iloc[:180]
    full,fw=walk_forward(prices,cfg)
    short,sw=walk_forward(shortened,cfg)
    common=full.index.isin(short.index)
    a=full[common].sort_values(['strategy','signal_date'])
    b=short.sort_values(['strategy','signal_date'])
    assert np.allclose(a.net_return,b.net_return)
    assert np.allclose(fw[fw.index.isin(sw.index)].sort_values(['strategy','date']).iloc[:,1:],sw.sort_values(['strategy','date']).iloc[:,1:])

def test_csv_input_validation(tmp_path):
    path=tmp_path/'prices.csv'
    path.write_text('date,a,b\n2020-01-01,100,90\n2020-01-01,100,90\n')
    with pytest.raises(ValueError):load_prices(path)

def test_invalid_cov_and_views():
    with pytest.raises(ValueError):black_litterman(np.array([[1,2],[2,1]]),np.array([.5,.5]))
    with pytest.raises(ValueError):black_litterman(COV,MARKET,P=np.eye(3),Q=np.ones(3),omega=np.zeros((3,3)))
    with pytest.raises(ValueError):optimal_weights(np.zeros(3),COV,max_weight=.2)

def test_metrics_zero_and_cli(tmp_path):
    assert metrics(np.zeros(20))['sharpe']==0.
    main(['--days','170','--lookback','65','--output',str(tmp_path)])
    for name in ('returns.csv','weights.csv','metrics.json','equity_curves.png'):
        assert (tmp_path/name).exists()
