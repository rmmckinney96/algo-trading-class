import logging
from typing import List, Optional, Type
from datetime import datetime, timedelta

from trading_strategy.models.instrument import Instrument
from trading_strategy.models.market_data import MarketData
from trading_strategy.models.strategy_config import StrategyConfig
from trading_strategy.models.order import Order, ExecutionResult
from trading_strategy.models.performance import PerformanceMetrics
from trading_strategy.core.base_strategy import BaseStrategy

class ExecutionManager:
    def __init__(
        self, 
        strategy_class: Type[BaseStrategy],
        instrument: Instrument,
        config: StrategyConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Execution Manager with a specific strategy.
        
        :param strategy_class: The trading strategy class to be used
        :param instrument: The financial instrument to trade
        :param config: Strategy configuration
        :param logger: Optional custom logger
        """
        self.strategy_class = strategy_class
        self.instrument = instrument
        self.config = config
        
        # Initialize strategy
        self.strategy = strategy_class(
            instrument=instrument, 
            config=config
        )
        
        # Logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Execution tracking
        self.total_executions: List[ExecutionResult] = []
        self.current_execution: Optional[Order] = None
        
        # Performance aggregation
        self.performance_metrics = PerformanceMetrics()
        
        # Risk management trackers
        self._weekly_trade_count = 0
        self._last_trade_timestamp: Optional[datetime] = None
        self._trading_halted = False
        self._halt_start_time: Optional[datetime] = None

    def process_market_data(self, market_data: List[MarketData]):
        """
        Process a series of market data points.
        
        :param market_data: List of market data points to process
        """
        for data_point in market_data:
            try:
                # Check global risk management constraints
                if not self._check_risk_constraints(data_point):
                    continue
                
                # Execute strategy logic
                self.strategy.process_market_data(data_point)
                
                # Update tracking and logging
                self._update_trade_tracking(data_point)
                
            except Exception as e:
                self.logger.error(f"Error processing market data: {e}")
                # Optionally add more detailed error handling

    def _check_risk_constraints(self, market_data: MarketData) -> bool:
        """
        Enforce global risk management constraints.
        
        :param market_data: Current market data point
        :return: Boolean indicating if trading is allowed
        """
        current_time = market_data.timestamp
        
        # Weekly trade limit
        if self._weekly_trade_count >= self.config.max_weekly_trades:
            self.logger.warning("Weekly trade limit reached")
            return False
        
        # Trading halt management
        if self._trading_halted:
            halt_duration = current_time - self._halt_start_time
            if halt_duration < timedelta(hours=self.config.trading_halt_period_hours):
                return False
            else:
                # Reset halt status after waiting period
                self._trading_halted = False
                self._halt_start_time = None
        
        # Drawdown protection
        current_equity = self.strategy.current_capital
        peak_equity = self.strategy.peak_capital
        
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown > self.config.drawdown_limit:
                self.logger.warning(f"Drawdown limit exceeded: {drawdown * 100:.2f}%")
                self._initiate_trading_halt(current_time)
                return False
        
        return True

    def _update_trade_tracking(self, market_data: MarketData):
        """
        Update trade tracking and performance metrics.
        
        :param market_data: Current market data point
        """
        # Track weekly trades
        if (self._last_trade_timestamp is None or 
            market_data.timestamp.isocalendar()[1] != 
            self._last_trade_timestamp.isocalendar()[1]):
            # Reset weekly trade count on new week
            self._weekly_trade_count = 0
        
        # Update trade count if a new trade was executed
        if self.strategy.current_position:
            self._weekly_trade_count += 1
            self._last_trade_timestamp = market_data.timestamp

    def _initiate_trading_halt(self, current_time: datetime):
        """
        Halt trading due to risk management triggers.
        
        :param current_time: Timestamp of halt initiation
        """
        self._trading_halted = True
        self._halt_start_time = current_time
        self.logger.warning("Trading halted due to risk management")

    def get_performance_summary(self) -> PerformanceMetrics:
        """
        Aggregate and return performance metrics.
        
        :return: Comprehensive performance metrics
        """
        # Combine strategy-level and execution-level metrics
        performance = self.strategy.performance
        
        # Additional aggregations can be added here
        performance.total_return = (
            self.strategy.current_capital / self.config.initial_capital - 1
        ) * 100
        
        return performance

    def export_trade_log(self, file_path: Optional[str] = None):
        """
        Export trade execution history.
        
        :param file_path: Optional file path to export trade log
        """
        trade_log = self.strategy.execution_history
        
        if file_path:
            try:
                import pandas as pd
                df = pd.DataFrame([
                    {
                        'symbol': trade.order.symbol,
                        'entry_price': trade.order.entry_price,
                        'exit_price': trade.exit_price,
                        'profit_loss': trade.profit_loss,
                        'entry_time': trade.order.timestamp,
                        'exit_time': trade.exit_time,
                        'reason': trade.reason
                    } for trade in trade_log
                ])
                df.to_csv(file_path, index=False)
                self.logger.info(f"Trade log exported to {file_path}")
            except ImportError:
                self.logger.error("Pandas not installed. Cannot export trade log.")
        
        return trade_log

    def visualize_performance(self):
        """
        Generate performance visualization.
        Placeholder for more advanced visualization logic.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Plot equity curve
            plt.figure(figsize=(10, 6))
            equity_curve = self.strategy.equity_curve
            plt.plot([
                entry['timestamp'] for entry in equity_curve
            ], [
                entry['equity'] for entry in equity_curve
            ])
            plt.title(f"Equity Curve - {self.strategy_class.__name__}")
            plt.xlabel("Time")
            plt.ylabel("Equity")
            plt.tight_layout()
            plt.show()
        except ImportError:
            self.logger.warning("Matplotlib not installed. Cannot visualize performance.")

# Example usage in a script
def main():
    # Import necessary components
    from trading_strategy.models.instrument import Instrument
    from trading_strategy.models.strategy_config import StrategyConfig
    from trading_strategy.strategies.trend_following import TrendFollowingStrategy
    from trading_strategy.data.data_loader import MarketDataLoader

    # Configure instrument
    instrument = Instrument(
        symbol="USD/JPY", 
        base_currency="USD", 
        quote_currency="JPY"
    )

    # Configure strategy
    config = StrategyConfig(
        name="Trend Following Strategy",
        initial_capital=10000,
        max_position_risk=0.02,
        max_weekly_trades=3
    )

    # Load market data
    market_data = MarketDataLoader.load_historical_data(instrument)

    # Create execution manager
    execution_manager = ExecutionManager(
        strategy_class=TrendFollowingStrategy,
        instrument=instrument,
        config=config
    )

    # Process market data
    execution_manager.process_market_data(market_data)

    # Get performance summary
    performance = execution_manager.get_performance_summary()
    print(performance)

    # Export trade log
    execution_manager.export_trade_log("trade_log.csv")

    # Visualize performance
    execution_manager.visualize_performance()

if __name__ == "__main__":
    main()
