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
DEFAULT_SYMBOL = "BTC-USD"
PERIOD = "3d"
INTERVAL = "15m"
REFRESH_MS = 5_000
RSI_LENGTH = 14

LINE_TOLERANCE_PCT = 0.0001   # 0.015% of current price
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
    Flat-to-flat model with alternating direction:

      Trade 1: BUY -> SELL (long)
      Trade 2: SELL -> BUY (short)
      Trade 3: BUY -> SELL (long)
      ...

    Each click adds a level (or deletes if within tolerance).
    PnL is computed from completed entry/exit pairs.
    If there's an open trade (odd number of levels), we compute unrealized PnL.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol

        self.fig, (self.ax_price, self.ax_rsi) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 6)
        )
        self.fig.subplots_adjust(hspace=0.05)

        self.price_line = None
        self.rsi_line = None
        self.df = None

        # Clicked price levels (in order)
        self.levels: list[float] = []
        self.level_artists = []

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.update_data_and_redraw()

        timer = self.fig.canvas.new_timer(interval=REFRESH_MS)
        timer.add_callback(self.update_data_and_redraw)
        timer.start()

    # ---------- Data ----------
    def fetch_data(self):
        df = yf.download(
            self.symbol,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
        )
        #print(df)
        if df.empty:
            print(f"No data for {self.symbol}")
            return df

        df["RSI"] = compute_rsi(df["Close"], RSI_LENGTH)
        return df.dropna()

    def get_last_price(self):
        if self.price_line is None:
            return None
        y = np.asarray(self.price_line.get_ydata())
        if len(y) == 0:
            return None
        return float(y[-1])

    # ---------- Alternating flat direction ----------
    def side_for_level_index(self, i: int) -> str:
        #print("side dor level index")
        """
        i = 0-based index into self.levels
        Even-indexed levels are ENTRIES, odd-indexed are EXITS.

        Direction alternates per trade:
          trade_idx = i // 2
          trade 0 entry = BUY (long)
          trade 1 entry = SELL (short)
          trade 2 entry = BUY (long)
          ...

        Entry side:
          BUY if trade_idx even else SELL
        Exit side is opposite of entry.
        """
        trade_idx = i // 2
        is_entry = (i % 2 == 0)
        #print("trade_idx:", trade_idx)
        entry_side = "SELL" if (trade_idx % 2 == 0) else "BUY"
        if is_entry:
            return entry_side
        return "SELL" if entry_side == "SELL" else "BUY"

    def simulate(self, last_price: float):
        """
        Computes:
          realized, unrealized, total, status_string

        Realized from completed pairs (entry, exit).
        Unrealized from open trade if odd number of levels.
        """
        realized = 0.0
        unrealized = 0.0
        status = "Risk Off"

        n = len(self.levels)
    
        pairs = n // 2

        # Realized for each completed trade
        for t in range(pairs):
            entry = self.levels[2*t]
            exit_ = self.levels[2*t + 1]

            entry_side = "BUY" if (t % 2 == 0) else "SELL"

            if entry_side == "BUY":
                realized += (exit_ - entry)      # long
            else:
                realized += (entry - exit_)      # short

        # Open trade?
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
            print("unrealized:", unrealized)    

        total = (realized + unrealized)
        return realized, unrealized, total, status

    # ---------- Drawing ----------
    def update_data_and_redraw(self):
        df = self.fetch_data()
        if df.empty:
            return

        self.df = df
        x = df.index
        p = df["Close"]
        r = df["RSI"]

        if self.price_line is None:
            self.price_line, = self.ax_price.plot(x, p, lw=.5, color = "blue")
        else:
            self.price_line.set_data(x, p)
        if self.rsi_line is None:
            self.rsi_line, = self.ax_rsi.plot(x, r, lw=.25)
        else:
            self.rsi_line.set_data(x, r)

        last_price = float(p.iloc[-1])
        self.rsi_line, = self.ax_rsi.plot(x, r, lw=.25, color="black")
        realized, unrealized, total, status = self.simulate(last_price)

 #       self.ax_price.set_title("Main Title", fontsize=14, pad=20)

        self.ax_price.text(
            0.5, 1.09,
            "Click To Trade Using RSI",
            transform=self.ax_price.transAxes,
            ha="center",
            va="bottom",
            fontsize = 16,
            color="black"
        )
        self.ax_price.set_title(
            f"{self.symbol}  "
            f"{last_price:,.3f}  "
            f"Interval:{INTERVAL}  "
            f"Realized:{realized:,.3f}  "
            f"Unrealized:{unrealized:,.3f}  "
            f"Total:{total:,.3f}  "
            f"{status}",
            color="black",
            fontsize = 15

        )        
        self.ax_price.set_ylabel("Price")
        
        self.ax_price.set_facecolor("lightgray")
        self.ax_rsi.set_facecolor("cyan")

        self.ax_price.relim()
        self.ax_price.autoscale_view()

        self.ax_rsi.set_ylabel("RSI")
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.axhline(30, ls="--", alpha=0.25)
        self.ax_rsi.axhline(70, ls="--", alpha=0.25)

        self.ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        self.fig.autofmt_xdate()
        #print(type(self.ax_price))
        #print(dir(self.ax_price))
        self.redraw_levels()
        self.fig.canvas.draw_idle()
    def color_from_trade_count(self, trade_count: int) -> str:
        #cycle = ["green", "black", "red", "black"]
        cycle = ["black", "green", "black", "red"]

        return cycle[trade_count % 4]
    def redraw_levels(self):
        for a in self.level_artists:
            a.remove()
        self.level_artists = []
        count = 0   
        # Draw lines; (keeping style simple—semantics are in title & PnL)
        #Direction alternates per trade:
        #trade_idx = i // 2
        #trade 0 entry = BUY (long)
        #trade 1 entry = SELL (short)
        #trade 2 entry = BUY (long)
        for lvl in self.levels:
            count = count + 1
            color = self.color_from_trade_count(count)

            line = self.ax_price.axhline(lvl, ls="--", alpha=0.85, color=color)
            self.level_artists.append(line)

    # ---------- Toggle line ----------
    def toggle_level(self, price_value: float):
        last_price = self.get_last_price()
        if last_price is None:
            return
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!last_price:", last_price)
        tol = last_price * LINE_TOLERANCE_PCT

        # Find nearest existing level within tolerance
        nearest_idx = None
        nearest_delta = None
        for i, lvl in enumerate(self.levels):
            d = abs(lvl - price_value)
            #print("d:",d)
            if d <= tol and (nearest_delta is None or d < nearest_delta):
                nearest_delta = d
                nearest_idx = i

        # Delete if near
        #if nearest_idx is not None:
            #del self.levels[nearest_idx]
            #self.redraw_levels()
            #self.fig.canvas.draw_idle()
            #return

        # Otherwise add
        self.levels.append(float(price_value))
        self.redraw_levels()
        self.fig.canvas.draw_idle()

    # ---------- Click handlers ----------
    def on_click(self, event):
        if event.button != 1:
            return

        if event.inaxes == self.ax_price and event.ydata is not None:
            self.toggle_level(float(event.ydata))
            return

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

        self.toggle_level(float(row["Close"]))


def main():
    sym = input(f"Ticker (default {DEFAULT_SYMBOL}): ").strip() or DEFAULT_SYMBOL
    print(f"Using {sym}")
    print("Flat-to-flat alternating model:")
    print("Trade 1: BUY→SELL (long), Trade 2: SELL→BUY (short), etc.")
    print("Click near an existing line to delete it.")
    TickerWithRSIPlot(sym)
    plt.show()


if __name__ == "__main__":
    main()