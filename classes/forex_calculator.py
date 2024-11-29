class ForexCalculator:
    @staticmethod
    def calculate_profit_usd(entry_price, exit_price, position_size):
        pip_value = 0.01
        pips_gained = (exit_price - entry_price) / pip_value
        profit = (pips_gained * pip_value) * position_size
        return profit

    @staticmethod
    def calculate_trailing_stop_price(highest_price, trailing_stop_percentage):
        return highest_price * (1 - trailing_stop_percentage)