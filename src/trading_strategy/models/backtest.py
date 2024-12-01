from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel

from trading_strategy.models.position import Position
from trading_strategy.models.equity import EquityPoint
from trading_strategy.models.strategy_config import StrategyConfig
from trading_strategy.models.market_data import MarketData
from trading_strategy.strategies.trend_following import TrendFollowingStrategy
from trading_strategy.indicators.technical import calculate_indicators_vectorized

class TradeMetrics(BaseModel):
    """Statistics for a set of trades"""
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_trade_return: float
    avg_duration: timedelta
    sharpe_ratio: float
    avg_daily_return: float

class BacktestResults(BaseModel):
    """Complete results of a strategy backtest"""
    strategy_config: StrategyConfig
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: List[Position]
    equity_curve: List[EquityPoint]
    metrics: TradeMetrics
    market_data: List[MarketData]
    
    @property
    def duration(self) -> timedelta:
        """Total duration of backtest"""
        return self.end_date - self.start_date
    
    @property
    def annualized_return(self) -> float:
        """Calculate annualized return"""
        years = self.duration.days / 365
        return ((1 + self.total_return_pct/100) ** (1/years) - 1) * 100 if years > 0 else 0

class BacktestRunner(BaseModel):
    """Configures and runs strategy backtests"""
    strategy_config: StrategyConfig
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    transaction_cost_pct: float = 0.001  # 0.1% per trade
    spread_cost_pips: float = 2.0  # Typical USDJPY spread
    financing_rate: float = 0.02  # 2% annual financing cost
    
    def calculate_transaction_costs(self, position_size: float) -> float:
        """Calculate total transaction costs including spread and commission"""
        commission = abs(position_size) * self.transaction_cost_pct
        spread_cost = abs(position_size) * (self.spread_cost_pips / 100)  # Convert pips to price
        return commission + spread_cost
    
    def calculate_financing_cost(self, position_size: float, hold_time_days: float) -> float:
        """Calculate financing costs for holding positions"""
        daily_rate = self.financing_rate / 365
        financing_cost = position_size * daily_rate * hold_time_days
        return financing_cost
    
    def run(self, market_data: List[MarketData]) -> BacktestResults:
        """Run backtest and return results"""
        # Initialize strategy
        strategy = TrendFollowingStrategy(self.strategy_config)
        
        # Filter data by date range if specified
        if self.start_date:
            market_data = [d for d in market_data if d.timestamp >= self.start_date]
        if self.end_date:
            market_data = [d for d in market_data if d.timestamp <= self.end_date]
            
        # Run strategy
        for data in market_data:
            # Calculate indicators
            data_with_indicators = strategy.calculate_indicators(data)
            
            # Check for position exit
            if strategy.position and strategy.check_exit_conditions(data_with_indicators):
                strategy.close_position(data_with_indicators)
            
            # Check for position entry
            elif not strategy.position and strategy.check_entry_conditions(data_with_indicators):
                position_size = strategy.calculate_position_size(data_with_indicators)
                strategy.open_position(data_with_indicators, position_size)
        
        # Calculate metrics
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'size': t.size,
            'pnl': t.pnl,
            'return': (t.pnl / (t.entry_price * abs(t.size))) if t.size != 0 else 0
        } for t in strategy.trades])
        
        if len(trades_df) > 0:
            trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
            trades_df['return_per_day'] = trades_df['return'] / (trades_df['duration'].dt.total_seconds() / (24*3600))
            
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                              trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            
            metrics = TradeMetrics(
                total_trades=len(trades_df),
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_pnl=trades_df['pnl'].mean(),
                avg_trade_return=trades_df['return'].mean() * 100,
                avg_duration=trades_df['duration'].mean(),
                sharpe_ratio=trades_df['return'].mean() / trades_df['return'].std() if trades_df['return'].std() != 0 else 0,
                avg_daily_return=trades_df['return_per_day'].mean() * 100
            )
        else:
            metrics = TradeMetrics(
                total_trades=0,
                win_rate=0,
                profit_factor=0,
                avg_trade_pnl=0,
                avg_trade_return=0,
                avg_duration=timedelta(0),
                sharpe_ratio=0,
                avg_daily_return=0
            )
        
        # Calculate drawdown
        equity_series = pd.Series([p.equity for p in strategy.equity_curve])
        drawdowns = (equity_series.cummax() - equity_series) / equity_series.cummax() * 100
        max_drawdown = drawdowns.max()
        
        return BacktestResults(
            strategy_config=self.strategy_config,
            start_date=market_data[0].timestamp,
            end_date=market_data[-1].timestamp,
            initial_capital=self.strategy_config.initial_capital,
            final_equity=strategy.current_equity,
            total_return_pct=(strategy.current_equity / self.strategy_config.initial_capital - 1) * 100,
            max_drawdown_pct=max_drawdown,
            trades=strategy.trades,
            equity_curve=strategy.equity_curve,
            metrics=metrics,
            market_data=market_data
        )
    
    def process_batch(self, market_data_batch: List[MarketData], batch_size: int = 1000) -> None:
        """Process market data in batches for better memory efficiency"""
        for i in range(0, len(market_data_batch), batch_size):
            batch = market_data_batch[i:i + batch_size]
            # Calculate indicators for batch
            indicators = calculate_indicators_vectorized(
                self.price_history[-100:] + batch,  # Keep some history for calculations
                self.strategy.indicator_periods
            )
            
            # Update market data with indicators
            for market_data, indicator_values in zip(batch, indicators):
                market_data.indicators.update(indicator_values)
                self.process_market_data(market_data)