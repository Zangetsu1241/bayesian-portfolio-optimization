"""Stable posterior estimates and constrained mean-variance allocation."""
import numpy as np
from scipy.optimize import minimize

TRADING_DAYS = 252

def covariance_estimate(returns, shrinkage=0.2, floor=1e-8):
    """Annualized sample covariance with diagonal shrinkage and eigenvalue floor."""
    x = np.asarray(returns, float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite T-by-N return matrix with T>=3, N>=2')
    if not 0 <= shrinkage <= 1 or floor <= 0:
        raise ValueError('Invalid covariance regularization')
    raw = np.cov(x, rowvar=False) * TRADING_DAYS
    cov = (1-shrinkage)*raw + shrinkage*np.diag(np.diag(raw))
    eigen, vec = np.linalg.eigh(cov)
    return (vec * np.maximum(eigen, floor)) @ vec.T

def _validate_cov(cov):
    cov = np.asarray(cov, float)
    if (cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2 or
        not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-10) or
        np.linalg.eigvalsh(cov).min() <= 0):
        raise ValueError('Covariance must be symmetric and positive definite')
    return cov

def black_litterman(cov, market_weights, risk_aversion=3., tau=0.05,
                     P=None, Q=None, omega=None):
    """Gaussian equilibrium prior on annual returns updated with uncertain views."""
    cov = _validate_cov(cov)
    n = len(cov)
    w = np.asarray(market_weights, float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError('market weights must be long-only and sum to one')
    if not np.isfinite(risk_aversion) or risk_aversion <= 0 or not np.isfinite(tau) or tau <= 0:
        raise ValueError('risk aversion and tau must be positive')
    pi = risk_aversion*cov@w
    prior_uncertainty = tau*cov
    if P is None and Q is None and omega is None:
        return pi, prior_uncertainty
    if any(item is None for item in (P, Q, omega)):
        raise ValueError('Supply P, Q, and omega together')
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    omega = np.asarray(omega, float)
    k = P.shape[0] if P.ndim == 2 else 0
    if (k == 0 or P.shape[1:] != (n,) or Q.shape != (k,) or
        omega.shape != (k,k) or not np.isfinite(P).all() or not np.isfinite(Q).all() or
        not np.isfinite(omega).all()):
        raise ValueError('Invalid view dimensions or values')
    _validate_view_cov(omega)
    innovation_cov = P@prior_uncertainty@P.T + omega
    gain = np.linalg.solve(innovation_cov, P@prior_uncertainty).T
    posterior_mean = pi+gain@(Q-P@pi)
    posterior_cov = prior_uncertainty - gain@P@prior_uncertainty
    posterior_cov = 0.5*(posterior_cov+posterior_cov.T)
    return posterior_mean, posterior_cov

def _validate_view_cov(c):
    if not np.allclose(c,c.T,atol=1e-10) or np.linalg.eigvalsh(c).min()<=0:
        raise ValueError('View covariance omega must be SPD')

def bayesian_mean(returns, covariance, prior_mean=None, prior_strength=20.):
    """Gaussian conjugate update; annualized posterior mean and mean uncertainty."""
    x = np.asarray(returns,float)
    cov = _validate_cov(covariance)
    if x.ndim != 2 or x.shape[1] != cov.shape[0] or len(x)<3 or not np.isfinite(x).all():
        raise ValueError('Invalid return observations')
    if not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError('prior_strength must be positive')
    prior = np.zeros(cov.shape[0]) if prior_mean is None else np.asarray(prior_mean,float)
    if prior.shape != (cov.shape[0],) or not np.isfinite(prior).all():
        raise ValueError('Invalid prior mean')
    annual_sample=x.mean(axis=0)*TRADING_DAYS
    mu=(prior_strength*prior+len(x)*annual_sample)/(prior_strength+len(x))
    mu_uncertainty=cov*TRADING_DAYS/(prior_strength+len(x))
    return mu,mu_uncertainty

def optimal_weights(mu, cov, risk_aversion=3., max_weight=1.):
    """Maximize expected annual utility, fully invested, long-only, capped weights."""
    mu=np.asarray(mu,float)
    cov=_validate_cov(cov)
    n=len(cov)
    if mu.shape != (n,) or not np.isfinite(mu).all():
        raise ValueError('Expected returns do not match covariance')
    if (not np.isfinite(risk_aversion) or risk_aversion <= 0 or
        not 1/n <= max_weight <= 1):
        raise ValueError('Invalid risk aversion or position cap')
    def objective(w): return -float(mu@w-0.5*risk_aversion*w@cov@w)
    def gradient(w): return -(mu-risk_aversion*cov@w)
    result=minimize(objective,np.ones(n)/n,jac=gradient,method='SLSQP',
         bounds=[(0.,max_weight)]*n,
         constraints=[{'type':'eq','fun':lambda w: w.sum()-1.,'jac':lambda w: np.ones_like(w)}],
         options={'ftol':1e-12,'maxiter':1000})
    if not result.success or not np.isclose(result.x.sum(),1,atol=1e-6) or np.any(result.x < -1e-7):
        raise RuntimeError('Portfolio optimization failed: '+str(result.message))
    return np.clip(result.x,0,max_weight)/np.clip(result.x,0,max_weight).sum()

def minimum_variance(cov,max_weight=1.):
    cov=_validate_cov(cov)
    return optimal_weights(np.zeros(len(cov)),cov,max_weight=max_weight)

def default_absolute_views(returns, cov, tau=0.05, uncertainty_multiple=2., lookback=21):
    """Momentum-derived absolute views, estimated only from trailing observations."""
    x=np.asarray(returns,float)
    cov=_validate_cov(cov)
    if x.ndim != 2 or x.shape[1] != len(cov) or len(x)<lookback:
        raise ValueError('Insufficient history for views')
    if uncertainty_multiple<=0 or not np.isfinite(uncertainty_multiple):
        raise ValueError('Invalid view uncertainty')
    P=np.eye(len(cov))
    Q=x[-lookback:].mean(axis=0)*TRADING_DAYS
    omega=np.diag(np.maximum(np.diag(tau*cov)*uncertainty_multiple,1e-10))
    return P,Q,omega
