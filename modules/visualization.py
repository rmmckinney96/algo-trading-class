import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_returns(strategy_returns, portfolio_returns):
    """Plot cumulative returns for strategies and portfolio"""
    plt.figure(figsize=(12, 6))
    ((1 + strategy_returns).cumprod()).plot(
        title='Cumulative Strategy Returns'
    )
    ((1 + portfolio_returns).cumprod()).plot(
        linewidth=3, 
        color='black', 
        label='Portfolio'
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def plot_correlation_matrix(strategy_returns):
    """Plot correlation matrix for strategy returns"""
    plt.figure(figsize=(10, 8))
    corr_matrix = strategy_returns.corr()
    
    im = plt.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    plt.xticks(np.arange(len(corr_matrix.columns)), 
               corr_matrix.columns, 
               rotation=45, 
               ha='right')
    plt.yticks(np.arange(len(corr_matrix.columns)), 
               corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha="center", va="center", color="black")
    
    plt.title('Strategy Correlations')
    plt.tight_layout()
    return plt.gcf()

def plot_rolling_volatility(strategy_returns):
    """Plot rolling annualized volatility"""
    plt.figure(figsize=(12, 6))
    strategy_returns.rolling(252).std().mul(np.sqrt(252)).plot(
        title='Rolling Annualized Volatility'
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def plot_weights_history(weights_history):
    """Plot strategy weights over time"""
    plt.figure(figsize=(12, 6))
    weights_history.plot(
        title='Strategy Weights Over Time'
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf() 