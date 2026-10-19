"""Trading Intelligence Platform — Native Qt Interactive Candlestick & VWAP Chart Widget.

Renders live intraday OHLCV candlesticks, session VWAP curve, and Entry / Stop Loss / Target
price levels directly via QPainter without any browser or WebView dependency.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class LiveCandlestickChartWidget(QWidget):
    """Native Windows Qt chart widget for live intraday candles + Entry/SL/Target levels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(270)
        self.setMouseTracking(True)
        self._candles: list[dict[str, Any]] = []
        self._entry: float = 0.0
        self._stop: float = 0.0
        self._target: float = 0.0
        self._signal: str = "NO-TRADE"
        self._symbol: str = "NSE:TCS"
        self._currency: str = "INR"
        self._market: str = "NSE"
        self._status_message: str = ""
        self._hover_pos: QPointF | None = None

    def update_chart_data(
        self,
        candles: list[dict[str, Any]],
        symbol: str,
        signal: str,
        entry: float,
        stop: float,
        target: float,
        currency: str = "INR",
        market: str = "NSE",
        status_message: str = "",
    ) -> None:
        """Incrementally update chart state and trigger repaint only when needed."""
        self._candles = list(candles[-65:]) if candles else []
        self._symbol = symbol
        self._signal = signal
        self._entry = float(entry or 0.0)
        self._stop = float(stop or 0.0)
        self._target = float(target or 0.0)
        self._currency = currency
        self._market = market
        self._status_message = status_message
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._hover_pos = event.position()
        self.update()

    def leaveEvent(self, event: Any) -> None:
        self._hover_pos = None
        self.update()

    def _fmt_price(self, val: float) -> str:
        if self._market in ("FOREX", "QUOTEX") and val < 250.0 and self._currency != "INR":
            return f"{val:,.5f}"
        if self._currency == "INR" or self._market in ("NSE", "BSE"):
            return f"₹{val:,.2f}"
        if self._currency in ("USD", "USDT"):
            return f"${val:,.2f}"
        return f"{val:,.2f}"

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        painter.fillRect(rect, QColor("#0B1120"))

        # Border
        border_pen = QPen(QColor("#1E293B"), 1)
        painter.setPen(border_pen)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)

        if not self._candles:
            painter.setPen(QColor("#94A3B8"))
            font = QFont("Segoe UI", 12, QFont.Weight.DemiBold)
            painter.setFont(font)
            msg = self._status_message or f"No live chart candles available for {self._symbol}"
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, msg)
            painter.end()
            return

        left_pad = 18.0
        right_pad = 115.0
        top_pad = 32.0
        bottom_pad = 28.0

        plot_w = max(80.0, rect.width() - left_pad - right_pad)
        plot_h = max(80.0, rect.height() - top_pad - bottom_pad)
        plot_rect = QRectF(left_pad, top_pad, plot_w, plot_h)

        # Calculate min/max price including active Entry/SL/Target if within reasonable proximity
        highs = [float(c.get("high", 0.0)) for c in self._candles]
        lows = [float(c.get("low", 0.0)) for c in self._candles if float(c.get("low", 0.0)) > 0]
        if not highs or not lows:
            painter.end()
            return

        min_p = min(lows)
        max_p = max(highs)
        if self._signal in ("BUY", "SELL"):
            for lvl in (self._entry, self._stop, self._target):
                if lvl > 0 and min_p * 0.85 <= lvl <= max_p * 1.15:
                    min_p = min(min_p, lvl)
                    max_p = max(max_p, lvl)

        span = max(max_p - min_p, max_p * 0.002, 1e-5)
        min_p -= span * 0.08
        max_p += span * 0.08
        span = max_p - min_p

        def price_to_y(p: float) -> float:
            ratio = (p - min_p) / span
            return plot_rect.bottom() - ratio * plot_rect.height()

        # Draw horizontal grid lines + price labels
        grid_pen = QPen(QColor("#1E293B"), 1, Qt.PenStyle.DotLine)
        label_font = QFont("Segoe UI", 8)
        painter.setFont(label_font)
        steps = 5
        for i in range(steps + 1):
            p_val = min_p + (span * i / steps)
            y = price_to_y(p_val)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))
            painter.setPen(QColor("#64748B"))
            painter.drawText(
                QRectF(plot_rect.right() + 6, y - 9, right_pad - 10, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._fmt_price(p_val),
            )

        # Header caption inside chart
        header_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(header_font)
        painter.setPen(QColor("#94A3B8"))
        painter.drawText(
            QRectF(left_pad, 6, plot_w, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"LIVE INTRADAY PRICE & SESSION VWAP — {self._symbol} ({len(self._candles)} bars)",
        )

        # Draw Candlesticks & compute running session VWAP
        n = len(self._candles)
        step_x = plot_w / max(n, 1)
        body_w = max(3.0, min(14.0, step_x * 0.68))

        cum_pv = 0.0
        cum_v = 0.0
        vwap_points: list[QPointF] = []
        hovered_idx: int | None = None

        for idx, c in enumerate(self._candles):
            o_p = float(c.get("open", 0.0))
            h_p = float(c.get("high", o_p))
            l_p = float(c.get("low", o_p))
            c_p = float(c.get("close", o_p))
            vol = max(1.0, float(c.get("volume", 1.0)))

            cx = plot_rect.left() + (idx + 0.5) * step_x
            if self._hover_pos and abs(self._hover_pos.x() - cx) <= step_x * 0.5:
                hovered_idx = idx

            tp = (h_p + l_p + c_p) / 3.0
            cum_pv += tp * vol
            cum_v += vol
            vwap_val = cum_pv / cum_v
            vwap_points.append(QPointF(cx, price_to_y(vwap_val)))

            is_bull = c_p >= o_p
            color = QColor("#10B981") if is_bull else QColor("#EF4444")

            # Wick
            painter.setPen(QPen(color, 1.3))
            painter.drawLine(QPointF(cx, price_to_y(h_p)), QPointF(cx, price_to_y(l_p)))

            # Body
            y_open = price_to_y(o_p)
            y_close = price_to_y(c_p)
            top_y = min(y_open, y_close)
            body_h = max(2.0, abs(y_close - y_open))
            body_rect = QRectF(cx - body_w / 2.0, top_y, body_w, body_h)
            painter.fillRect(body_rect, color)

        # Draw VWAP line
        if len(vwap_points) >= 2:
            vwap_path = QPainterPath(vwap_points[0])
            for pt in vwap_points[1:]:
                vwap_path.lineTo(pt)
            painter.setPen(QPen(QColor("#F59E0B"), 1.8))
            painter.drawPath(vwap_path)

        # Draw Entry, Stop Loss, and Target overlays when BUY or SELL
        def draw_level_line(price_val: float, label: str, hex_color: str) -> None:
            if price_val <= 0 or not (min_p <= price_val <= max_p):
                return
            y_lvl = price_to_y(price_val)
            col = QColor(hex_color)
            pen = QPen(col, 1.4, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(plot_rect.left(), y_lvl), QPointF(plot_rect.right(), y_lvl))

            tag_rect = QRectF(plot_rect.right() + 4, y_lvl - 9, right_pad - 8, 18)
            painter.fillRect(tag_rect, col)
            painter.setPen(QColor("#0F172A"))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(
                tag_rect,
                Qt.AlignmentFlag.AlignCenter,
                f"{label} {self._fmt_price(price_val)}",
            )

        if self._signal in ("BUY", "SELL"):
            draw_level_line(self._entry, "ENTRY", "#38BDF8")
            draw_level_line(self._stop, "STOP", "#F87171")
            draw_level_line(self._target, "TARGET", "#34D399")

        # Hover inspection bar
        if hovered_idx is not None and 0 <= hovered_idx < len(self._candles):
            hc = self._candles[hovered_idx]
            ts_str = str(hc.get("timestamp", ""))[-14:-6] or str(hc.get("timestamp", ""))[:16]
            info = (
                f"Bar [{ts_str}]   O: {self._fmt_price(float(hc.get('open', 0)))}   "
                f"H: {self._fmt_price(float(hc.get('high', 0)))}   "
                f"L: {self._fmt_price(float(hc.get('low', 0)))}   "
                f"C: {self._fmt_price(float(hc.get('close', 0)))}"
            )
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QColor("#E2E8F0"))
            painter.drawText(
                QRectF(left_pad, rect.height() - 22, plot_w, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                info,
            )

        painter.end()
