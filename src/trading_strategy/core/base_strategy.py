from abc import ABC, abstractmethod
from typing import List, Optional

from trading_strategy.models.instrument import Instrument
from trading_strategy.models.market_data import MarketData
from trading_strategy.models.order import Order, ExecutionResult
from trading_strategy.models.performance import PerformanceMetrics
from trading_strategy.models.strategy_config import StrategyConfig

class BaseStrategy(ABC):
    def __init__(
        self, 
        instrument: Instrument, 
        config: StrategyConfig
    ):
        self.instrument = instrument
        self.config = config
        
        self.current_capital = config.initial_capital
        self.peak_capital = config.initial_capital
        
        self.current_position: Optional[Order] = None
        self.performance = PerformanceMetrics()
        self.execution_history: List[ExecutionResult] = []
        self.equity_curve: List[dict] = []

    @abstractmethod
    def calculate_position_size(self, market_data: MarketData) -> float:
        """Calculate position size based on risk management."""
        pass

    @abstractmethod
    def check_entry_conditions(self, market_data: MarketData) -> bool:
        """Determine if entry conditions are met."""
        pass

    @abstractmethod
    def check_exit_conditions(self, market_data: MarketData) -> bool:
        """Determine if exit conditions are met."""
        pass

    def open_position(self, market_data: MarketData) -> Order:
        """Open a new trading position."""
        position_size = self.calculate_position_size(market_data)
        
        order = Order(
            symbol=self.instrument.symbol,
            side='buy',  # Can be overridden in specific strategies
            quantity=position_size,
            entry_price=market_data.close_price
        )
        
        self.current_position = order
        return order

    def close_position(
        self, 
        market_data: MarketData, 
        reason: str = 'strategy_exit'
    ) -> ExecutionResult:
        """Close the current trading position."""
        if not self.current_position:
            raise ValueError("No open position to close")
        
        execution_result = ExecutionResult(
            order=self.current_position,
            exit_price=market_data.close_price,
            exit_time=market_data.timestamp,
            profit_loss=self._calculate_trade_pnl(
                self.current_position.entry_price, 
                market_data.close_price, 
                self.current_position.quantity
            ),
            reason=reason
        )
        
        # Update performance metrics
        self._update_performance_metrics(execution_result)
        
        self.execution_history.append(execution_result)
        self.current_position = None
        
        return execution_result

    def _calculate_trade_pnl(
        self, 
        entry_price: float, 
        exit_price: float, 
        position_size: float
    ) -> float:
        """Calculate profit and loss for a trade."""
        pip_value = 0.01
        pips_gained = (exit_price - entry_price) / pip_value
        return pips_gained * pip_value * position_size

    def _update_performance_metrics(self, execution_result: ExecutionResult):
        """Update performance metrics after trade execution."""
        self.performance.total_trades += 1
        if execution_result.profit_loss > 0:
            self.performance.winning_trades += 1
        else:
            self.performance.losing_trades += 1
        
        # Update capital
        self.current_capital += execution_result.profit_loss

    def update_equity_curve(self, market_data: MarketData):
        """Update equity curve tracking."""
        current_equity = self.current_capital
        
        # Include unrealized PnL if position is open
        if self.current_position:
            unrealized_pnl = self._calculate_trade_pnl(
                self.current_position.entry_price, 
                market_data.close_price, 
                self.current_position.quantity
            )
            current_equity += unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': market_data.timestamp,
            'equity': current_equity
        })
        
        # Update peak capital and max drawdown
        self.peak_capital = max(self.peak_capital, current_equity)
        self.performance.max_drawdown = max(
            self.performance.max_drawdown, 
            (self.peak_capital - current_equity) / self.peak_capital
        )

    def process_market_data(self, market_data: MarketData):
        """Main trading logic processor."""
        # Update equity curve
        self.update_equity_curve(market_data)
        
        # Manage existing position
        if self.current_position:
            if self.check_exit_conditions(market_data):
                self.close_position(market_data)
            return

        # Check for new entry
        if self.check_entry_conditions(market_data):
            self.open_position(market_data)
