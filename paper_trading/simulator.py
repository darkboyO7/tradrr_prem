"""Trading Intelligence Platform - Execution Simulator.

Simulates exchange matching engine behavior, bid/ask spread crossing,
taker/maker fee schedules, and market impact slippage.
"""


from paper_trading.models import OrderSide, OrderType, PaperOrder


class ExecutionSimulator:
    """Simulates realistic execution fills with spread and slippage penalties."""

    def __init__(
        self,
        half_spread_bps: float = 1.0,  # 1 basis point (0.01%) half-spread
        taker_fee_pct: float = 0.0006,  # 6 bps taker fee
        maker_fee_pct: float = 0.0002,  # 2 bps maker fee
        slippage_base_bps: float = 1.5,  # 1.5 bps baseline slippage
    ) -> None:
        self.half_spread_bps = half_spread_bps
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct
        self.slippage_base_bps = slippage_base_bps

    def simulate_fill(
        self,
        order: PaperOrder,
        market_price: float,
    ) -> tuple[float, float, float]:
        """Calculate realistic fill price, fee amount, and total slippage in price units.

        Returns:
            Tuple of (fill_price, fee_paid, slippage_amount)
        """
        half_spread = (self.half_spread_bps / 10000.0) * market_price
        base_slippage = (self.slippage_base_bps / 10000.0) * market_price

        if order.order_type == OrderType.MARKET:
            fee_rate = self.taker_fee_pct
            if order.side == OrderSide.BUY:
                fill_price = market_price + half_spread + base_slippage
                slippage = half_spread + base_slippage
            else:
                fill_price = market_price - half_spread - base_slippage
                slippage = half_spread + base_slippage
        else:
            # Passive limit order - executes at limit price without adverse slippage
            fee_rate = self.maker_fee_pct
            fill_price = order.limit_price if order.limit_price is not None else market_price
            slippage = 0.0

        notional = fill_price * order.quantity
        fee = notional * fee_rate

        return round(fill_price, 4), round(fee, 4), round(slippage, 4)
