"""Command line Bayesian portfolio research experiment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import synthetic_prices,load_prices
from .backtest import Config,walk_forward,metrics

def main(argv=None):
    parser=argparse.ArgumentParser(description='Bayesian and Black-Litterman portfolio research')
    parser.add_argument('--csv',default=None,help='date and 2+ adjusted daily asset prices')
    parser.add_argument('--days',type=int,default=850)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--lookback',type=int,default=126)
    parser.add_argument('--rebalance-every',type=int,default=21)
    parser.add_argument('--cost-bps',type=float,default=5.)
    parser.add_argument('--output',default='results')
    args=parser.parse_args(argv)
    prices=load_prices(args.csv) if args.csv else synthetic_prices(args.days,args.seed)
    cfg=Config(lookback=args.lookback,rebalance_every=args.rebalance_every,cost_bps=args.cost_bps)
    trades,weights=walk_forward(prices,cfg)
    reports={name:metrics(frame.net_return,frame.turnover) for name,frame in trades.groupby('strategy')}
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    trades.to_csv(out/'returns.csv')
    weights.to_csv(out/'weights.csv')
    summary={'dataset':'user-provided' if args.csv else 'simulated',
             'parameters':vars(args),'portfolio_settings':cfg.__dict__,
             'limitations':'Synthetic demo; closing signal at t cannot execute retroactively at close t; commission-only cost model.',
             'metrics':reports}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(figsize=(11,6))
    for name,frame in trades.groupby('strategy'):
        wealth=(1+frame.net_return).cumprod()
        ax.plot(wealth.index,wealth,label=name)
    ax.set(title='Portfolio growth (illustrative, cost-adjusted)',ylabel='Wealth from $1')
    ax.legend(loc='best',fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'equity_curves.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'dataset':summary['dataset'],'metrics':reports},indent=2))
    return summary

if __name__=='__main__':main()
