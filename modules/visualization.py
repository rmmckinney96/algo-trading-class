import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from datetime import datetime

def plot_strategy_comparison(portfolio, figsize: tuple = (12, 6)):
    """Plot cumulative returns for all strategies and portfolio"""
    plt.figure(figsize=figsize)
    
    # Plot individual strategy returns
    ((1 + portfolio.returns.drop('Portfolio', axis=1)).cumprod()).plot(
        style='--', 
        alpha=0.5
    )
    
    # Plot portfolio returns
    ((1 + portfolio.returns['Portfolio']).cumprod()).plot(
        linewidth=3, 
        color='black', 
        label='Portfolio'
    )
    
    plt.title('Cumulative Strategy Returns')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_portfolio_metrics(portfolio, lookback_days: int = 252):
    """Plot various portfolio metrics"""
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Rolling Volatility
    returns = portfolio.returns
    rolling_vol = returns.rolling(lookback_days).std() * np.sqrt(24 * 252)  # Annualized from hourly
    rolling_vol.plot(
        ax=axes[0,0], 
        title='Rolling Annualized Volatility'
    )
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].set_ylabel('Volatility')
    axes[0,0].grid(True)

    # 2. Strategy Correlations
    corr_matrix = returns.drop('Portfolio', axis=1).corr()
    im = axes[0,1].imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    axes[0,1].set_xticks(np.arange(len(corr_matrix.columns)))
    axes[0,1].set_yticks(np.arange(len(corr_matrix.columns)))
    axes[0,1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    axes[0,1].set_yticklabels(corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = axes[0,1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black")
    axes[0,1].set_title('Strategy Correlations')
    plt.colorbar(im, ax=axes[0,1])

    # 3. Rolling Sharpe Ratio
    rolling_ret = returns.rolling(lookback_days).mean() * (24 * 252)  # Annualized
    rolling_vol = returns.rolling(lookback_days).std() * np.sqrt(24 * 252)
    rolling_sharpe = rolling_ret / rolling_vol
    rolling_sharpe.plot(
        ax=axes[1,0], 
        title='Rolling Sharpe Ratio'
    )
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,0].grid(True)

    # 4. Strategy Weights Over Time
    weights = pd.DataFrame(portfolio.allocations, index=[0])
    weights.plot(
        kind='bar',
        ax=axes[1,1],
        title='Strategy Allocations'
    )
    axes[1,1].set_ylabel('Weight')
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_drawdowns(portfolio, n_worst: int = 5):
    """Plot worst drawdowns"""
    returns = portfolio.returns['Portfolio']
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    
    plt.figure(figsize=(12, 6))
    drawdowns.plot()
    plt.title(f'Portfolio Drawdowns')
    plt.ylabel('Drawdown')
    plt.grid(True)
    
    # Find worst drawdowns
    drawdown_periods = []
    current_drawdown = 0
    start_idx = None
    
    for i, dd in enumerate(drawdowns):
        if dd < 0 and current_drawdown == 0:
            start_idx = i
        current_drawdown = min(current_drawdown, dd)
        if dd == 0 and start_idx is not None:
            drawdown_periods.append({
                'start': drawdowns.index[start_idx],
                'end': drawdowns.index[i],
                'drawdown': current_drawdown
            })
            current_drawdown = 0
            start_idx = None
    
    # Sort and print worst drawdowns
    drawdown_periods.sort(key=lambda x: x['drawdown'])
    print(f"\nWorst {n_worst} Drawdowns:")
    for i, dd in enumerate(drawdown_periods[:n_worst]):
        print(f"{i+1}. {dd['drawdown']:.2%} ({dd['start']} to {dd['end']})")
    
    plt.tight_layout()
    plt.show()

def plot_strategy_analysis(portfolio, 
                          strategy_name: Optional[str] = None,
                          start_date: Optional[Union[str, datetime]] = None,
                          end_date: Optional[Union[str, datetime]] = None,
                          figsize: tuple = (15, 10)):
    """
    Detailed analysis of a specific strategy or the whole portfolio
    
    Parameters:
    -----------
    portfolio : Portfolio
        Portfolio object containing strategy returns
    strategy_name : str, optional
        Name of specific strategy to analyze. If None, analyzes portfolio
    start_date, end_date : str or datetime, optional
        Date range for analysis
    """
    # Filter returns by date if specified
    returns = portfolio.returns
    if start_date:
        returns = returns[returns.index >= pd.to_datetime(start_date)]
    if end_date:
        returns = returns[returns.index <= pd.to_datetime(end_date)]
    
    # Select strategy or portfolio returns
    if strategy_name:
        if strategy_name not in returns.columns:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        series_to_analyze = returns[strategy_name]
        title_prefix = f"Strategy: {strategy_name}"
    else:
        series_to_analyze = returns['Portfolio']
        title_prefix = "Portfolio"

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Cumulative Returns
    cum_returns = (1 + series_to_analyze).cumprod()
    cum_returns.plot(ax=axes[0,0], title=f'{title_prefix} Cumulative Returns')
    axes[0,0].set_ylabel('Cumulative Return')
    axes[0,0].grid(True)

    # 2. Monthly Returns Heatmap
    monthly_returns = series_to_analyze.groupby([
        series_to_analyze.index.year,
        series_to_analyze.index.month
    ]).sum().unstack()
    im = axes[0,1].imshow(monthly_returns, cmap='RdYlGn')
    axes[0,1].set_title(f'{title_prefix} Monthly Returns')
    plt.colorbar(im, ax=axes[0,1])
    
    # 3. Rolling Metrics
    rolling_window = 24*30  # 30 days for hourly data
    rolling_ret = series_to_analyze.rolling(rolling_window).mean() * (24 * 252)
    rolling_vol = series_to_analyze.rolling(rolling_window).std() * np.sqrt(24 * 252)
    rolling_sharpe = rolling_ret / rolling_vol
    
    ax2 = axes[1,0].twinx()
    rolling_ret.plot(ax=axes[1,0], color='blue', label='Return')
    rolling_vol.plot(ax=ax2, color='red', label='Volatility')
    axes[1,0].set_title(f'{title_prefix} Rolling Metrics (30d)')
    axes[1,0].grid(True)
    
    # 4. Drawdown Analysis
    cum_returns = (1 + series_to_analyze).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    drawdowns.plot(ax=axes[1,1], title=f'{title_prefix} Drawdowns')
    axes[1,1].set_ylabel('Drawdown')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{title_prefix} Summary Statistics:")
    stats = {
        'Total Return': (cum_returns.iloc[-1] - 1),
        'Ann. Return': (1 + series_to_analyze).prod() ** (252*24/len(series_to_analyze)) - 1,
        'Ann. Volatility': series_to_analyze.std() * np.sqrt(24 * 252),
        'Sharpe Ratio': (series_to_analyze.mean() * (24 * 252)) / (series_to_analyze.std() * np.sqrt(24 * 252)),
        'Max Drawdown': drawdowns.min(),
        'Win Rate': (series_to_analyze > 0).mean(),
        'Avg Win': series_to_analyze[series_to_analyze > 0].mean(),
        'Avg Loss': series_to_analyze[series_to_analyze < 0].mean(),
    }
    
    for metric, value in stats.items():
        if 'Rate' in metric:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.4f}")