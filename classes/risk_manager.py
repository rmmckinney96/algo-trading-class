class RiskManager:
    def __init__(self, initial_capital, drawdown_limit):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.drawdown_limit = drawdown_limit # DRAWDOWN_LIMIT
        
    def update_peak_capital(self, current_capital):
        self.peak_capital = max(self.peak_capital, current_capital)
        
    def check_drawdown(self, current_capital):
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        return drawdown <= self.drawdown_limit
        
    def reset_peak_capital(self, current_equity):
        self.peak_capital = current_equity
    
    def get_minimum_equity(self):
        return self.peak_capital * (1 - self.drawdown_limit)