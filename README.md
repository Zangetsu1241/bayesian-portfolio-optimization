# Bayesian Portfolio Optimization Research

A reproducible **Black-Litterman and Bayesian mean estimation** research project with cost-adjusted chronological backtesting.

## Research question
Do posterior estimates of expected returns and estimation uncertainty improve outcomes against classical sample-mean optimization, equal-weight and minimum-variance benchmarks? Results vary with prior, market, and costs.

## Setup
Requires Python 3.10+.
```bash
python -m pip install -e '.[test]'
python -m pytest -q
bayes-portfolio --days 850 --output results
```
Use a `date` column followed by at least two positive, point-in-time adjusted asset close columns:
```bash
bayes-portfolio --csv adjusted_prices.csv --lookback 126 --rebalance-every 21 --cost-bps 5 --output market-results
```
Outputs: `returns.csv`, `weights.csv`, `metrics.json`, `equity_curves.png`.

## Models
- Equal weight, long-only minimum variance
- Classical trailing sample mean, optimized with annualized covariance
- Conjugate Gaussian Bayesian posterior using an equilibrium prior and known covariance approximation
- Black-Litterman equilibrium-implied prior
- Black-Litterman with momentum-derived, explicitly uncertain absolute views

Black-Litterman prior: `pi = delta * Sigma @ market_weights`. Mean prior covariance: `tau * Sigma`. View model: `Q = P @ mu + noise`; `noise ~ N(0, Omega)`. Posterior is computed using linear algebra solves, not explicit inversion.

Long-only optimizer maximizes `w @ mu - delta/2 * w @ Sigma_predictive @ w` with full investment and per-asset caps. Bayesian predictive covariances include posterior mean uncertainty.

## Experimental protocol
- Four synthetic correlated assets with switching market volatility and drift.
- Default 126-day rolling window, rebalance every 21 sessions.
- Only data observed on or before decision date are used for allocation.
- Assets drift between scheduled rebalances. Trading costs are proportional to turnover.
- All six strategies are compared on the same test dates.
- 15 tests cover posterior identities, confidence changes, SPD covariance, portfolio constraints, backtest prefix invariance, accounting, and CLI.

## Synthetic benchmark (seed 42, 850 trading days)

| Strategy | Total return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Bayesian mean | +90.70% | 1.306 | -12.73% |
| Sample mean | +74.98% | 0.995 | -17.14% |
| Equal weight | +39.97% | 0.779 | -25.03% |
| Black-Litterman prior | +38.32% | 0.771 | -24.49% |
| Black-Litterman views | +39.02% | 0.597 | -27.80% |
| Minimum variance | +6.16% | 0.246 | -14.56% |

**These results are synthetic**, not historical market evidence or a claim of live tradability. The Black-Litterman market portfolio is approximated by equal weights. Ex-ante momentum views are simplistic, and their uncertainty is a modeling assumption.

## Limitations
A signal computed after close t cannot execute at that same close. This close-to-close research simulator requires realistic next-open/MOC execution modeling before trading. Spreads, impact, financing, taxes, point-in-time universe construction, survivorship bias and market microstructure are not fully modeled. More markets and historical periods are needed for a reliable scientific comparison.

## Layout
`src/bayes_alloc/models.py`: Bayesian updates, views and optimization.
`src/bayes_alloc/data.py`: synthetic market, CSV validation.
`src/bayes_alloc/backtest.py`: rolling walk-forward research.
`src/bayes_alloc/cli.py`: experiment outputs.
`tests/test_models.py`: numerical and integration tests.
`.github/workflows/tests.yml`: CI on Python 3.10 and 3.12.

MIT licensed.
