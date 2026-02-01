# ---------- IMPORTANT: Fix macOS + Python 3.13 mouse events ----------
import matplotlib
matplotlib.use("TkAgg")  # REQUIRED FIX for mouse clicks on macOS/Py3.13
# --------------------------------------------------------------------

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field

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

LINE_TOLERANCE_PCT = 0.0001   # tolerance for deleting a nearby level (fraction of last price)
RSI_CLICK_TOL = 5.0           # +/- RSI points tolerance to accept RSI-curve click

APP_NAME = "rsi_price6"


# ===================== State (JSON) =====================

def default_state_path() -> Path:
    # macOS/Linux: ~/.config/<app>/state.json ; Windows: %APPDATA%\<app>\state.json
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_NAME / "state.json"


@dataclass
class AppState:
    symbol: str = DEFAULT_SYMBOL
    period: str = PERIOD
    interval: str = INTERVAL  # persisted field, but we'll NOT change it for the session override
    price_lines: list[float] = field(default_factory=list)


def load_state() -> AppState:
    path = default_state_path()
    if not path.exists():
        return AppState()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AppState()

    st = AppState()
    for k, v in raw.items():
        if hasattr(st, k):
            setattr(st, k, v)

    if not isinstance(st.price_lines, list):
        st.price_lines = []

    return st


def save_state(state: AppState) -> None:
    path = default_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    tmp.replace(path)


# ===================== Indicators =====================

def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ===================== App =====================

class TickerWithRSIPlot:
    """
    Flat-to-flat alternating model:
      Trade 1: BUY→SELL (long)
      Trade 2: SELL→BUY (short)
      Trade 3: BUY→SELL (long)
      ...

    Click price chart:
      - Adds a horizontal level
      - If click is near an existing level (within tolerance), deletes it

    Click RSI chart (near the RSI curve):
      - Toggles a level at that candle's Close

    Key:
      - 'r' resets all levels (and persists)
    """

    def __init__(self, symbol: str, state: AppState, interval: str):
        self.symbol = symbol
        self.state = state

        # SESSION-ONLY interval (NOT persisted)
        self.interval = interval

        self.fig, (self.ax_price, self.ax_rsi) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 6)
        )
        self.fig.subplots_adjust(hspace=0.05)

        self.df: pd.DataFrame | None = None

        # Restore saved levels
        self.levels: list[float] = list(self.state.price_lines)
        self.level_artists = []

        # Event hooks
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)  # press 'r'

        # Initial draw + timer refresh
        self.update_data_and_redraw()
        timer = self.fig.canvas.new_timer(interval=REFRESH_MS)
        timer.add_callback(self.update_data_and_redraw)
        timer.start()

    # ---------- Data ----------
    def fetch_data(self) -> pd.DataFrame:
        df = yf.download(
            self.symbol,
            period=self.state.period,
            interval=self.interval,   # <-- session-only interval
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        df["RSI"] = compute_rsi(df["Close"], RSI_LENGTH)
        return df.dropna()

    def last_price(self) -> float | None:
        if self.df is None or self.df.empty:
            return None
        return float(self.df["Close"].iloc[-1])

    # ---------- Trading model ----------
    def simulate(self, last_price: float):
        realized = 0.0
        unrealized = 0.0
        status = "Risk Off"

        n = len(self.levels)
        pairs = n // 2

        # Completed trades
        for t in range(pairs):
            entry = self.levels[2 * t]
            exit_ = self.levels[2 * t + 1]
            entry_side = "BUY" if (t % 2 == 0) else "SELL"
            realized += (exit_ - entry) if entry_side == "BUY" else (entry - exit_)

        # Open trade (odd number of levels)
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

    # ---------- Drawing ----------
    def color_from_trade_count(self, count: int) -> str:
        cycle = ["black", "green", "black", "red"]
        return cycle[count % 4]

    def redraw_levels(self):
        for a in self.level_artists:
            try:
                a.remove()
            except Exception:
                pass
        self.level_artists = []

        count = 0
        for lvl in self.levels:
            count += 1
            color = self.color_from_trade_count(count)
            line = self.ax_price.axhline(lvl, ls="--", alpha=0.85, color=color)
            self.level_artists.append(line)

    def update_data_and_redraw(self):
        df = self.fetch_data()
        if df.empty:
            return

        self.df = df
        x = df.index
        p = df["Close"]
        r = df["RSI"]

        last = float(p.iloc[-1])
        realized, unrealized, total, status = self.simulate(last)

        self.ax_price.clear()
        self.ax_rsi.clear()

        self.ax_price.plot(x, p, lw=0.5, color="blue")
        self.ax_rsi.plot(x, r, lw=0.25, color="black")

        self.ax_price.set_facecolor("lightgray")
        self.ax_rsi.set_facecolor("cyan")

        self.ax_price.text(
            0.5, 1.09,
            "Click To Trade Using RSI   (press 'r' to reset)",
            transform=self.ax_price.transAxes,
            ha="center",
            va="bottom",
            fontsize=16,
            color="black"
        )

        self.ax_price.set_title(
            f"{self.symbol}  {last:,.3f}  "
            f"Interval:{self.interval}  "
            f"Realized:{realized:,.3f}  "
            f"Unrealized:{unrealized:,.3f}  "
            f"Total:{total:,.3f}  {status}",
            color="black",
            fontsize=15
        )
        self.ax_price.set_ylabel("Price")

        self.ax_rsi.set_ylabel("RSI")
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.axhline(30, ls="--", alpha=0.25)
        self.ax_rsi.axhline(70, ls="--", alpha=0.25)

        self.ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        self.fig.autofmt_xdate()

        self.redraw_levels()
        self.fig.canvas.draw_idle()

    # ---------- Persistence helpers ----------
    def sync_state(self):
        self.state.price_lines = list(self.levels)

    # ---------- Toggle line ----------
    def toggle_level(self, price_value: float):
        last = self.last_price()
        if last is None:
            return

        tol = last * LINE_TOLERANCE_PCT

        nearest_idx = None
        nearest_delta = None
        for i, lvl in enumerate(self.levels):
            d = abs(lvl - price_value)
            if d <= tol and (nearest_delta is None or d < nearest_delta):
                nearest_delta = d
                nearest_idx = i

        if nearest_idx is not None:
            del self.levels[nearest_idx]
        else:
            self.levels.append(float(price_value))

        self.sync_state()
        self.redraw_levels()
        self.fig.canvas.draw_idle()

    # ---------- Events ----------
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

    def on_key(self, event):
        # Keypresses require figure focus: click once inside the chart, then press 'r'
        if event.key == "r":
            print("RESET (r)")
            self.levels.clear()
            self.sync_state()
            save_state(self.state)  # persist reset immediately
            self.redraw_levels()
            self.fig.canvas.draw_idle()


# ===================== Main =====================

def main():
    state = load_state()
    print("Loaded:", state)

    sym = input(f"Ticker (default {state.symbol}): ").strip() or state.symbol
    state.symbol = sym

    # interval override at startup (NOT persisted)
    interval_default = state.interval or INTERVAL
    interval = input(f"Interval (default {interval_default}): ").strip() or interval_default

    print("Using", sym)
    print("Session interval:", interval, "(not persisted)")
    print("Flat-to-flat alternating model:")
    print("Trade 1: BUY→SELL (long), Trade 2: SELL→BUY (short), etc.")
    print("Click near an existing line to delete it.")
    print("Tip: click inside the chart window once so keypresses work, then press 'r' to reset.")

    app = TickerWithRSIPlot(sym, state, interval)

    def _on_close(_evt):
        # interval override is session-only, and we never modified state.interval, so save is safe
        save_state(state)
        print("State saved:", default_state_path())

    app.fig.canvas.mpl_connect("close_event", _on_close)
    plt.show()


if __name__ == "__main__":
    main()