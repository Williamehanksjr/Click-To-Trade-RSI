Click-To-Trade-RSI
One-click RSI-based trade targeting with instant price feedback.

Designed by William E. Hanks
<img width="1440" height="787" alt="Screenshot 2026-02-06 at 5 03 29 PM" src="https://github.com/user-attachments/assets/7ca42e49-b62a-4e51-9f72-cad8143225b3" />
Click To Trade RSI

Overview

Click To Trade RSI is a lightweight, interactive day-trading simulator built with Python and Matplotlib.
It allows traders to execute BUY / SELL / EXIT (risk-off) actions using simple mouse clicks directly on Price or RSI charts, while tracking realized and unrealized P&L in real time.

The app is designed for discretionary, click-based trading workflows rather than automated execution.

⸻

Features
	•	Two synchronized charts
	•	Top: Price
	•	Bottom: RSI (14-period)
	•	Click-based trading workflow
	•	Sequential actions per click:
	1.	BUY (enter long)
	2.	EXIT (risk off)
	3.	SELL (enter short)
	4.	EXIT (risk off)
	•	Sequentially labeled trade lines drawn on the clicked chart
	•	Editable Period and Interval in the title area
	•	Real-time P&L tracking
	•	Realized P&L
	•	Unrealized P&L
	•	Total P&L
	•	Trade state display: long, short, or risk off
	•	Works with stocks, ETFs, crypto (via yfinance)
	•	No broker connection (simulation only)

⸻

Interface Layout
	•	Title Area
	•	App name: Click To Trade RSI
	•	Editable fields:
	•	Period (e.g. 1d, 5d, 1mo)
	•	Interval (e.g. 1m, 5m, 15m)
	•	Top Plot
	•	Price chart
	•	Buy/Sell/Exit lines drawn at click location
	•	Bottom Plot
	•	RSI (0–100)
	•	Overbought / Oversold lines (70 / 30)
	•	Bottom Status Panel
	•	Traded symbol
	•	Current price
	•	Realized P&L
	•	Unrealized P&L
	•	Total P&L
	•	Purchase price
	•	Trade state


  
  
	•	BUY opens a long position at current market price
	•	SELL opens a short position at current market price
	•	EXIT closes any open position and realizes P&L
	•	Horizontal lines are drawn at the clicked Y-value and labeled sequentially

⸻

Data Source
	•	Market data fetched via yfinance
	•	Uses:
	•	Close price
	•	Auto-adjusted prices
	•	RSI calculated using Wilder’s smoothing (EMA-based)

⸻

Installation

⸻
