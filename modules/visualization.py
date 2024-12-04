import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from datetime import datetime

def plot_strategy_comparison(portfolio: 'Portfolio'):
    """Compare strategy performance"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot strategy values
    for name, strategy in portfolio.strategies.items():
        strategy.positions.groupby('timestamp')['portfolio_value'].first().plot(
            ax=ax1, label=name
        )
    
    ax1.set_title('Strategy Values')
    ax1.grid(True)
    ax1.legend()
    
    # Plot cumulative returns
    for name, strategy in portfolio.strategies.items():
        returns = strategy.positions.groupby('timestamp')['returns'].sum()
        (1 + returns).cumprod().plot(ax=ax2, label=name)
    
    ax2.set_title('Cumulative Returns')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_metrics(portfolio: 'Portfolio'):
    """Plot key portfolio metrics over time"""
    positions = portfolio.positions
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Portfolio Metrics', fontsize=16)
    
    # Portfolio Value
    positions.groupby('timestamp')['portfolio_value'].first().plot(ax=axes[0,0])
    axes[0,0].set_title('Portfolio Value')
    axes[0,0].grid(True)
    
    # PnL Components
    pnl_components = positions.groupby('timestamp').agg({
        'returns': 'sum',
        'costs': 'sum'
    })
    pnl_components.plot(ax=axes[0,1], kind='area', stacked=True)
    axes[0,1].set_title('PnL Components')
    axes[0,1].grid(True)
    
    # Returns Distribution
    positions.groupby('timestamp')['returns'].sum().hist(
        bins=50, ax=axes[1,0], density=True
    )
    axes[1,0].set_title('Returns Distribution')
    axes[1,0].grid(True)
    
    # Drawdowns
    portfolio_value = positions.groupby('timestamp')['portfolio_value'].first()
    drawdowns = (portfolio_value - portfolio_value.cummax()) / portfolio_value.cummax()
    drawdowns.plot(ax=axes[1,1])
    axes[1,1].set_title('Drawdowns')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_drawdowns(portfolio: 'Portfolio'):
    """Plot detailed drawdown analysis"""
    pnl_history = portfolio.pnl_history.set_index('timestamp')
    returns = pnl_history['portfolio_value'].pct_change()
    cumulative_returns = (1 + returns).cumprod()
    drawdowns = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    drawdowns.plot()
    plt.title('Portfolio Drawdowns')
    plt.grid(True)
    plt.ylabel('Drawdown %')
    
    # Add top 5 drawdowns annotation
    sorted_drawdowns = drawdowns.sort_values()
    top_5_drawdowns = sorted_drawdowns.head()
    for date, dd in top_5_drawdowns.items():
        plt.annotate(f'{dd:.1%}', 
                    xy=(date, dd),
                    xytext=(10, 10),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.show()

def plot_strategy_analysis(portfolio: 'Portfolio', strategy_name: str):
    """Plot detailed analysis for a specific strategy"""
    strategy = portfolio.strategies[strategy_name]
    positions = strategy.positions
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Strategy Analysis: {strategy_name}', fontsize=16)
    
    # Strategy Value
    positions.groupby('timestamp')['portfolio_value'].first().plot(ax=axes[0,0])
    axes[0,0].set_title('Strategy Value')
    axes[0,0].grid(True)
    
    # Position Sizes
    active_positions = positions[positions['is_active']]
    active_positions.pivot(
        index='timestamp', 
        columns='symbol', 
        values='size'
    ).plot(ax=axes[0,1])
    axes[0,1].set_title('Position Sizes')
    axes[0,1].grid(True)
    
    # Returns Distribution
    positions.groupby('timestamp')['returns'].sum().hist(
        bins=50, ax=axes[1,0], density=True
    )
    axes[1,0].set_title('Returns Distribution')
    axes[1,0].grid(True)
    
    # Rolling Metrics
    returns = positions.groupby('timestamp')['returns'].sum()
    rolling_window = 30
    rolling_sharpe = (
        returns.rolling(window=rolling_window).mean() / 
        returns.rolling(window=rolling_window).std() * 
        np.sqrt(252)
    )
    rolling_sharpe.plot(ax=axes[1,1])
    axes[1,1].set_title(f'Rolling {rolling_window}-Day Sharpe')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()