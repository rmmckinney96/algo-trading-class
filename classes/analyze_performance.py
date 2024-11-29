def analyze_performance(trade_log_df, strategy, equity_curve_df, data):
    
    trades = trade_log_df[trade_log_df['Type'] == 'Sell'].copy()
    equity_series = equity_curve_df['Equity']
    
    # Convert timestamps to datetime if they aren't already
    equity_curve_df['Date'] = pd.to_datetime(equity_curve_df['Date'])
    
    # Add month-end flag and calculate monthly returns
    equity_curve_df['YearMonth'] = equity_curve_df['Date'].dt.to_period('M')
    monthly_returns = equity_curve_df.groupby('YearMonth')['Equity'].agg(['first', 'last']).pct_change()
    monthly_returns = (monthly_returns['last'] * 100).dropna()
    
    # 1. Basic Performance Metrics
    final_capital = strategy.capital
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_trades = len(trades)
    profitable_trades = len(trades[trades['Profit'] > 0])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 2. Exit Analysis
    exit_analysis = trades.groupby('Exit_Reason').agg({
        'Profit': ['count', 'sum', 'mean', lambda x: (x > 0).sum() / len(x) * 100],
    }).round(2)
    
    exit_analysis.columns = ['Count', 'Total_Profit', 'Avg_Profit', 'Win_Rate_%']
      
    # 2. Time-Adjusted Return Metrics
    total_hours = (equity_curve_df['Date'].max() - equity_curve_df['Date'].min()).total_seconds() / 3600
    total_days = total_hours / 24
    total_years = total_days / 365
    
    # Monthly Return Statistics
    avg_monthly_return = monthly_returns.mean()
    monthly_return_std = monthly_returns.std()
    best_month = monthly_returns.max()
    worst_month = monthly_returns.min()
    positive_months = len(monthly_returns[monthly_returns > 0])
    negative_months = len(monthly_returns[monthly_returns < 0])
    total_months = len(monthly_returns)
    monthly_win_rate = (positive_months / total_months * 100) if total_months > 0 else 0
    
    # Annualized Return (adjusted for actual months)
    annual_return = ((1 + total_return/100) ** (1/total_years) - 1) * 100
    
    # Calculate hourly returns for intraday analysis
    equity_curve_df['Hourly_Returns'] = equity_curve_df['Equity'].pct_change()
    
    # 3. Risk Metrics
    hourly_volatility = equity_curve_df['Hourly_Returns'].std()
    daily_volatility = hourly_volatility * np.sqrt(24)
    annual_volatility = daily_volatility * np.sqrt(252)
    monthly_volatility = monthly_return_std
    
    # Drawdown Analysis
    equity_curve_df['Peak'] = equity_curve_df['Equity'].expanding().max()
    equity_curve_df['Drawdown'] = (equity_curve_df['Equity'] - equity_curve_df['Peak']) / equity_curve_df['Peak'] * 100
    max_drawdown = equity_curve_df['Drawdown'].min()
    avg_drawdown = equity_curve_df[equity_curve_df['Drawdown'] < 0]['Drawdown'].mean()
    
    # Drawdown duration
    drawdown_duration_hours = equity_curve_df[equity_curve_df['Drawdown'] < 0].groupby(
        (equity_curve_df['Drawdown'] >= 0).cumsum()
    ).size().max()
    
    # 4. Risk-Adjusted Returns
    annual_risk_free_rate = 0.02  # 2% annual
    monthly_risk_free_rate = (1 + annual_risk_free_rate) ** (1/12) - 1
    
    # Monthly Sharpe and Sortino Ratios
    monthly_excess_returns = monthly_returns/100 - monthly_risk_free_rate
    monthly_sharpe = monthly_excess_returns.mean() / monthly_excess_returns.std() * np.sqrt(12)
    
    negative_monthly_returns = monthly_returns[monthly_returns < 0]/100
    monthly_sortino = (monthly_returns.mean()/100 - monthly_risk_free_rate) / negative_monthly_returns.std() * np.sqrt(12)
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    # 5. Trade Statistics
    avg_profit = trades['Profit'].mean()
    median_profit = trades['Profit'].median()
    profit_std = trades['Profit'].std()
    largest_win = trades['Profit'].max()
    largest_loss = trades['Profit'].min()
    
    avg_win = trades[trades['Profit'] > 0]['Profit'].mean()
    avg_loss = trades[trades['Profit'] < 0]['Profit'].mean()
    win_loss_ratio = abs(avg_win/avg_loss) if avg_loss != 0 else np.inf
    
    gross_profits = trades[trades['Profit'] > 0]['Profit'].sum()
    gross_losses = abs(trades[trades['Profit'] < 0]['Profit'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
    
    print("\n" + "="*50)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("="*50)
    
    print("\n1. Monthly Return Statistics:")
    print(f"Average Monthly Return: {avg_monthly_return:.2f}%")
    print(f"Monthly Return Std Dev: {monthly_return_std:.2f}%")
    print(f"Best Month: {best_month:.2f}%")
    print(f"Worst Month: {worst_month:.2f}%")
    print(f"Positive Months: {positive_months} ({monthly_win_rate:.1f}%)")
    print(f"Negative Months: {negative_months}")
    print(f"Monthly Volatility: {monthly_volatility:.2f}%")
    
    print("\n2. Overall Performance Metrics:")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {annual_return:.2f}%")
    print(f"Total Months: {total_months}")
    print(f"Total Trades: {total_trades}")
    print(f"Trade Win Rate: {win_rate:.2f}%")
    
    print("\n3. Risk Metrics:")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Average Drawdown: {avg_drawdown:.2f}%")
    print(f"Longest Drawdown Duration: {drawdown_duration_hours:.1f} hours ({drawdown_duration_hours/24:.1f} days)")
    print(f"Annual Volatility: {annual_volatility*100:.2f}%")
    
    print("\n4. Risk-Adjusted Returns:")
    print(f"Monthly Sharpe Ratio (Annualized): {monthly_sharpe:.2f}")
    print(f"Monthly Sortino Ratio (Annualized): {monthly_sortino:.2f}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    print("\n5. Trade Statistics:")
    print(f"Average Profit per Trade: ${avg_profit:,.2f}")
    print(f"Median Profit per Trade: ${median_profit:,.2f}")
    print(f"Largest Win: ${largest_win:,.2f}")
    print(f"Largest Loss: ${largest_loss:,.2f}")
    print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")

    print("\n6. Exit Analysis:")
    print("\nExit Reason Analysis:")
    print(exit_analysis)
    print("\nDetailed Exit Analysis:")
    for reason in exit_analysis.index:
        total_exits = exit_analysis.loc[reason, 'Count']
        win_rate = exit_analysis.loc[reason, 'Win_Rate_%']
        avg_profit = exit_analysis.loc[reason, 'Avg_Profit']
        total_profit = exit_analysis.loc[reason, 'Total_Profit']
        
        print(f"\n{reason}:")
        print(f"  Number of exits: {total_exits} ({(total_exits/total_trades*100):.1f}% of all trades)")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Average profit: ${avg_profit:,.2f}")
        print(f"  Total profit contribution: ${total_profit:,.2f}")
    
    # Monthly returns visualization
    plt.figure(figsize=(15, 6))
    monthly_returns.plot(kind='bar')
    plt.title('Monthly Returns')
    plt.xlabel('Month')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Exit Analysis Visualization
    plt.figure(figsize=(12, 6))
    exit_counts = exit_analysis['Count'].plot(kind='bar')
    plt.title('Distribution of Exit Reasons')
    plt.xlabel('Exit Reason')
    plt.ylabel('Number of Trades')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for i, v in enumerate(exit_analysis['Count']):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Profit by Exit Reason Visualization
    plt.figure(figsize=(12, 6))
    exit_analysis['Avg_Profit'].plot(kind='bar')
    plt.title('Average Profit by Exit Reason')
    plt.xlabel('Exit Reason')
    plt.ylabel('Average Profit ($)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for i, v in enumerate(exit_analysis['Avg_Profit']):
        plt.text(i, v, f'${v:,.0f}', ha='center', va='bottom' if v >= 0 else 'top')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'avg_monthly_return': avg_monthly_return,
        'monthly_sharpe': monthly_sharpe,
        'monthly_sortino': monthly_sortino,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'monthly_win_rate': monthly_win_rate,
        'exit_analysis': exit_analysis
    }