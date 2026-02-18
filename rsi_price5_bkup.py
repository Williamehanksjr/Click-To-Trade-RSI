# ---------- IMPORTANT: Fix macOS + Python 3.13 mouse events ----------
import matplotlib
matplotlib.use("TkAgg")  # <-- REQUIRED FIX for mouse clicks on macOS/Py3.13
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# === CONFIG ===
DEFAULT_SYMBOL = "ETH-USD"   # use standard yfinance symbol format
PERIOD = "3d"
INTERVAL = "15m"
REFRESH_MS = 5_000
RSI_LENGTH = 14

LINE_TOLERANCE_PCT = 0.0001   # 0.01% of current price (used only if you re-enable delete)
RSI_CLICK_TOL = 5.0           # +/- 5 RSI points


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


class TickerWithRSIPlot:
    """
    Flat-to-flat alternating direction model:

      Trade 1: BUY -> SELL (long)
      Trade 2: SELL -> BUY (short)
      Trade 3: BUY -> SELL (long)
      ...

    Clicking adds a horizontal price level (trade marker).
    Press 'r' to reset (clear all trade lines).
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

        self.fig, (self.ax_price, self.ax_rsi) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 6)
        )
        self.fig.subplots_adjust(hspace=0.05)

        # main plotted lines (created once, then updated)
        self.price_line = None
        self.rsi_line = None

        # persistent header text (reuse; don't create on every refresh)
        self.header_text = None

        # RSI guide artists (created once so they don't stack)
        self.rsi_band = None
        self.rsi_os_line = None
        self.rsi_ob_line = None
        self.rsi_os_label = None
        self.rsi_ob_label = None

        self.df = None

        # Clicked price levels (in order)
        self.levels: list[float] = []
        self.level_artists = []

        # events
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # style backgrounds once
        self.ax_price.set_facecolor("lightgray")
        self.ax_rsi.set_facecolor("cyan")

        # create RSI guides ONCE (no stacking)
        self._init_rsi_guides()

        # initial draw
        self.update_data_and_redraw()

        # periodic refresh
        timer = self.fig.canvas.new_timer(interval=REFRESH_MS)
        timer.add_callback(self.update_data_and_redraw)
        timer.start()

    # ---------- RSI guide lines/band (created once) ----------
    def _init_rsi_guides(self):
        self.ax_rsi.set_ylim(0, 100)

        # Optional: fill the normal band
        self.rsi_band = self.ax_rsi.axhspan(30, 70, alpha=0.08)

        # Overbought / Oversold lines (more visible)
        self.rsi_os_line = self.ax_rsi.axhline(30, ls="--", lw=1.2, alpha=0.8)
        self.rsi_ob_line = self.ax_rsi.axhline(70, ls="--", lw=1.2, alpha=0.8)

        # Labels on the right
        self.rsi_os_label = self.ax_rsi.text(
            1.01, 30, "30",
            va="center",
            transform=self.ax_rsi.get_yaxis_transform()
        )
        self.rsi_ob_label = self.ax_rsi.text(
            1.01, 70, "70",
            va="center",
            transform=self.ax_rsi.get_yaxis_transform()
        )

    # ---------- Data ----------
    def fetch_data(self) -> pd.DataFrame:
        df = yf.download(
            self.symbol,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
        )
        if df.empty:
            print(f"No data for {self.symbol}")
            return df

        close = df["Close"]
        # yfinance can return Close as a 1-col DataFrame in some cases; flatten it
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.astype(float)

        df["RSI"] = compute_rsi(close, RSI_LENGTH)
        return df.dropna()

    def _close_series(self) -> pd.Series:
        """Return Close as a float Series (handles DataFrame/MultiIndex cases)."""
        if self.df is None or self.df.empty:
            return pd.Series(dtype=float)

        close = self.df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.astype(float)

    def get_last_price(self):
        if self.price_line is None:
            return None
        y = np.asarray(self.price_line.get_ydata())
        if len(y) == 0:
            return None
        return float(y[-1])

    # ---------- PnL ----------
    def simulate(self, last_price: float):
        realized = 0.0
        unrealized = 0.0
        status = "Risk Off"

        n = len(self.levels)
        pairs = n // 2

        for t in range(pairs):
            entry = self.levels[2*t]
            exit_ = self.levels[2*t + 1]
            entry_side = "BUY" if (t % 2 == 0) else "SELL"

            if entry_side == "BUY":
                realized += (exit_ - entry)  # long
            else:
                realized += (entry - exit_)  # short

        if n % 2 == 1:
            t = n // 2
            entry = self.levels[-1]
            entry_side = "BUY" if (t % 2 == 0) else "SELL"

            if entry_side == "BUY":
                unrealized = (last_price - entry)
                status = f"OPEN LONG @ {entry:.3f}"
            else:
                unrealized = (entry - last_price)
                status = f"OPEN SHORT @ {entry:.3f}"

        total = realized + unrealized
        return realized, unrealized, total, status

    # ---------- Reset ----------
    def reset(self):
        self.levels.clear()

        # remove level artists immediately
        for a in list(self.level_artists):
            try:
                a.remove()
            except Exception:
                pass
        self.level_artists.clear()

        print("RESET: cleared all levels")

        # force title refresh immediately so it looks like it worked
        self._update_titles_only()
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        k = (getattr(event, "key", "") or "").lower()
        if k == "r":
            self.reset()

    # ---------- Titles / header ----------
    def _update_titles_only(self):
        if self.df is None or self.df.empty:
            return

        close = self._close_series()
        if close.empty:
            return

        last_price = float(close.iloc[-1])
        realized, unrealized, total, status = self.simulate(last_price)

        header = "Click To Trade Using RSI  (r = reset)"
        if self.header_text is None:
            self.header_text = self.ax_price.text(
                0.5, 1.09,
                header,
                transform=self.ax_price.transAxes,
                ha="center",
                va="bottom",
                fontsize=16,
                color="black"
            )
        else:
            self.header_text.set_text(header)

        self.ax_price.set_title(
            f"{self.symbol}  "
            f"{last_price:,.3f}  "
            f"Interval:{INTERVAL}  "
            f"Realized:{realized:,.3f}  "
            f"Unrealized:{unrealized:,.3f}  "
            f"Total:{total:,.3f}  "
            f"{status}  |  r=reset",
            color="black",
            fontsize=15
        )

    # ---------- Drawing ----------
    def update_data_and_redraw(self):
        df = self.fetch_data()
        if df.empty:
            return

        self.df = df
        x = df.index

        # Close series (guaranteed Series of floats)
        p = self._close_series()
        if p.empty:
            return

        # RSI series (handle DataFrame too, just in case)
        r = df["RSI"]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[:, 0]
        r = r.astype(float)

        # ✅ Price plot line created once; updated thereafter (no duplicates)
        if self.price_line is None:
            self.price_line, = self.ax_price.plot(x, p, lw=0.5, color="blue")
        else:
            self.price_line.set_data(x, p)

        # ✅ RSI plot line created once; updated thereafter (no duplicates)
        if self.rsi_line is None:
            self.rsi_line, = self.ax_rsi.plot(x, r, lw=0.25, color="black")
        else:
            self.rsi_line.set_data(x, r)

        last_price = float(p.iloc[-1])
        realized, unrealized, total, status = self.simulate(last_price)

        # Header + Title (reused; no stacking)
        self._update_titles_only()

        # axes labels/scales
        self.ax_price.set_ylabel("Price")
        self.ax_rsi.set_ylabel("RSI")

        self.ax_price.relim()
        self.ax_price.autoscale_view()

        # x formatting
        self.ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        self.fig.autofmt_xdate()

        # redraw trade level lines (cleanly)
        self.redraw_levels()

        self.fig.canvas.draw_idle()

    def color_from_trade_count(self, trade_count: int) -> str:
        cycle = ["black", "green", "black", "red"]
        return cycle[trade_count % 4]

    def redraw_levels(self):
        # remove old artists
        for a in list(self.level_artists):
            try:
                a.remove()
            except Exception:
                pass
        self.level_artists.clear()

        # redraw all current levels
        count = 0
        for lvl in self.levels:
            count += 1
            color = self.color_from_trade_count(count)
            line = self.ax_price.axhline(lvl, ls="--", alpha=0.85, color=color)
            self.level_artists.append(line)

    # ---------- Toggle line ----------
    def toggle_level(self, price_value: float):
        # delete behavior currently disabled; always add
        self.levels.append(float(price_value))
        self.redraw_levels()
        self.fig.canvas.draw_idle()

    # ---------- Click handlers ----------
    def on_click(self, event):
        # Accept left clicks (or None button on some trackpad combos)
        if getattr(event, "button", None) is not None and event.button != 1:
            return

        # Price axis click -> trade level line at y
        if event.inaxes == self.ax_price and event.ydata is not None:
            self.toggle_level(float(event.ydata))
            return

        # RSI axis click -> nearest candle close if click is near RSI curve
        if event.inaxes == self.ax_rsi:
            self.handle_rsi_click(event)

    def handle_rsi_click(self, event):
        if self.df is None or self.df.empty:
            return
        if event.xdata is None or event.ydata is None:
            return

        xnum = mdates.date2num(self.df.index.to_pydatetime())
        idx = int(np.argmin(np.abs(xnum - event.xdata)))

        row = self.df.iloc[idx]
        rsi_here = float(row["RSI"])

        if abs(rsi_here - float(event.ydata)) > RSI_CLICK_TOL:
            return

        close = row["Close"]
        if isinstance(close, pd.Series):
            close = close.iloc[0]
        self.toggle_level(float(close))


def main():
    sym = input(f"Ticker (default {DEFAULT_SYMBOL}): ").strip() or DEFAULT_SYMBOL
    sym = sym.upper()
    print(f"Using {sym}")
    print("Press 'r' to reset (clear all trade lines).")
    TickerWithRSIPlot(sym)
    plt.show()


if __name__ == "__main__":
    main()