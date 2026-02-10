# ---------- IMPORTANT: Fix macOS + Python 3.13 mouse events ----------
import matplotlib
matplotlib.use("TkAgg")  # <-- REQUIRED FIX for mouse clicks on macOS/Py3.13
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Conditional CCXT import - only required if using Coinbase data source
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None


# === CONFIG ===
# Data source selector: Choose between Yahoo Finance and Coinbase
# Valid values: "yahoo" (default), "coinbase"
# 
# - "yahoo": Uses yfinance library (no additional dependencies)
# - "coinbase": Uses CCXT library to fetch data from Coinbase exchange
#               Requires: pip install ccxt
#
# To extend with other exchanges:
# 1. Add a new DATA_SOURCE option (e.g., "binance", "kraken")
# 2. Add corresponding logic in fetch_data() method
# 3. Use CCXT exchange-specific implementation (e.g., ccxt.binance())
# 4. Map symbol format as needed (each exchange has different formats)
DATA_SOURCE = "yahoo"  # Options: "yahoo", "coinbase"

# Symbol format depends on DATA_SOURCE:
# - Yahoo Finance: "BTC-USD", "ETH-USD", "AAPL", "SPY"
# - Coinbase: "BTC/USD", "ETH/USD"
# Update this symbol when changing DATA_SOURCE to match the expected format
DEFAULT_SYMBOL = "BTC-USD"  # For yahoo: "BTC-USD", For coinbase: "BTC/USD"
PERIOD = "3d"
INTERVAL = "15m"
REFRESH_MS = 5_000
RSI_LENGTH = 14

LINE_TOLERANCE_PCT = 0.0001   # 0.015% of current price
RSI_CLICK_TOL = 5.0           # +/- 5 RSI points


def map_interval_to_ccxt_timeframe(interval: str) -> str:
    """
    Maps yfinance interval format to CCXT timeframe format.
    
    yfinance uses: 1m, 5m, 15m, 30m, 1h, 1d, etc.
    CCXT uses: 1m, 5m, 15m, 30m, 1h, 1d, etc. (mostly compatible)
    
    Returns the CCXT-compatible timeframe string.
    
    NOTE: Some intervals may be mapped to different timeframes if not supported
    by CCXT. For example, 2m may fallback to 1m, and 90m to 1h. This ensures
    compatibility but may result in different data granularity than requested.
    """
    # Most intervals are already compatible between yfinance and CCXT
    # This mapping handles any edge cases
    interval_map = {
        "1m": "1m",
        "2m": "1m",  # Fallback to 1m if 2m not available
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "1h",
        "90m": "1h",  # Fallback to 1h
        "1h": "1h",
        "1d": "1d",
        "5d": "1d",  # Fallback to 1d
        "1wk": "1w",
        "1mo": "1M",
    }
    return interval_map.get(interval, interval)


def parse_period_to_days(period: str) -> int:
    """
    Converts yfinance period format to number of days.
    
    Examples: "1d" -> 1, "3d" -> 3, "1mo" -> 30, "1y" -> 365
    """
    period_map = {
        "1d": 1,
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "ytd": 365,  # Approximate
        "max": 3650,  # Limit to ~10 years
    }
    
    # Handle custom formats like "3d", "7d", etc.
    if period.endswith("d") and period[:-1].isdigit():
        return int(period[:-1])
    
    return period_map.get(period, 3)  # Default to 3 days


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
        """
        Fetches historical price data based on DATA_SOURCE configuration.
        
        Supported data sources:
        - "yahoo": Uses yfinance library (default)
        - "coinbase": Uses CCXT library to fetch from Coinbase exchange
        
        Returns a pandas DataFrame with DateTime index and columns:
        - Close: Closing price
        - RSI: Computed RSI values
        
        To add support for other exchanges (e.g., Binance, Kraken):
        1. Add elif block for new DATA_SOURCE value
        2. Initialize CCXT exchange: exchange = ccxt.binance()
        3. Fetch OHLCV data using exchange.fetch_ohlcv()
        4. Convert to DataFrame format as shown in coinbase example
        """
        if DATA_SOURCE == "yahoo":
            # Yahoo Finance data source (default)
            df = yf.download(
                self.symbol,
                period=PERIOD,
                interval=INTERVAL,
                progress=False,
            )
            if df.empty:
                print(f"No data for {self.symbol}")
                return df

            df["RSI"] = compute_rsi(df["Close"], RSI_LENGTH)
            return df.dropna()
        
        elif DATA_SOURCE == "coinbase":
            # Coinbase exchange data via CCXT
            # NOTE: This implementation uses public (unauthenticated) endpoints only.
            # Private endpoints (trading, account info) require API keys and are not
            # supported in this data-only implementation.
            if not CCXT_AVAILABLE:
                error_msg = (
                    f"ERROR: CCXT library not available.\n"
                    f"To use Coinbase data source, install CCXT:\n"
                    f"  pip install ccxt\n"
                    f"Then restart the application."
                )
                print(error_msg)
                return pd.DataFrame()
            
            try:
                # Initialize Coinbase exchange (no API keys - public endpoints only)
                exchange = ccxt.coinbase()
                
                # Map interval and period to CCXT format
                timeframe = map_interval_to_ccxt_timeframe(INTERVAL)
                days = parse_period_to_days(PERIOD)
                
                # Calculate start time (milliseconds since epoch)
                since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                
                # Calculate appropriate limit based on period and interval
                # This ensures we request enough data points to cover the period
                # CCXT typically allows up to 1000 candles per request
                # For 3 days at 15m intervals: 3 * 24 * 4 = 288 candles
                # For safety, we use a max of 1000 which covers most use cases
                limit = 1000  # Maximum data points (sufficient for most period/interval combos)
                
                # Fetch OHLCV data
                # Symbol format for Coinbase: "BTC/USD", "ETH/USD", etc.
                ohlcv = exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
                
                if not ohlcv:
                    print(f"No data received from Coinbase for {self.symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                # OHLCV format: [timestamp, open, high, low, close, volume]
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
                )
                
                # Convert timestamp to datetime index
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                
                # Compute RSI
                df["RSI"] = compute_rsi(df["Close"], RSI_LENGTH)
                
                return df.dropna()
                
            except ccxt.NetworkError as e:
                print(f"Network error fetching Coinbase data: {e}")
                return pd.DataFrame()
            except ccxt.ExchangeError as e:
                print(f"Coinbase exchange error: {e}")
                print(f"Note: Check if symbol '{self.symbol}' is valid for Coinbase.")
                print(f"Coinbase symbols use format: BTC/USD, ETH/USD, etc.")
                return pd.DataFrame()
            except Exception as e:
                print(f"Unexpected error fetching Coinbase data: {e}")
                return pd.DataFrame()
        
        else:
            # Unknown data source
            raise NotImplementedError(
                f"DATA_SOURCE='{DATA_SOURCE}' is not implemented.\n"
                f"Valid options: 'yahoo', 'coinbase'"
            )

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
