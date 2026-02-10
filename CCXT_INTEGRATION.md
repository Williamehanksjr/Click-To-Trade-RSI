# CCXT Integration

This document describes the CCXT integration added to the Click-To-Trade-RSI application.

## Overview

The application now supports fetching market data from multiple exchanges via the CCXT library, in addition to the existing yfinance integration.

## Configuration

### Data Source Selection

In `rsi_price5.py`, set the `DATA_SOURCE` variable:

```python
# Use yfinance (default, backward compatible)
DATA_SOURCE = "yfinance"

# OR use CCXT for exchange data
DATA_SOURCE = "ccxt"
```

### CCXT Configuration

When using `DATA_SOURCE = "ccxt"`, configure the following:

```python
EXCHANGE_NAME = "coinbase"  # Exchange to use (see supported exchanges below)
CCXT_LIMIT = 1000           # Number of candles to fetch
```

## Symbol Format

### yfinance Format
- Stocks: `AAPL`, `MSFT`, `TSLA`
- Crypto: `BTC-USD`, `ETH-USD`

### CCXT Format
- Spot trading: `BTC/USD`, `ETH/USDT`, `BTC/EUR`

**Auto-conversion**: The application automatically converts yfinance-style symbols (BTC-USD) to CCXT format (BTC/USD) when using CCXT.

## Supported Exchanges

CCXT supports 111+ exchanges including:
- Coinbase
- Binance
- Kraken
- Bitfinex
- Bitstamp
- And many more...

To see all available exchanges:
```python
import ccxt
print(ccxt.exchanges)
```

## Usage Examples

### Example 1: Using yfinance (Default)
```python
# In rsi_price5.py:
DATA_SOURCE = "yfinance"
DEFAULT_SYMBOL = "BTC-USD"

# Then run:
python rsi_price5.py
```

### Example 2: Using CCXT with Coinbase
```python
# In rsi_price5.py:
DATA_SOURCE = "ccxt"
EXCHANGE_NAME = "coinbase"
DEFAULT_SYMBOL = "BTC/USD"  # or "BTC-USD" (auto-converted)

# Then run:
python rsi_price5.py
```

### Example 3: Using CCXT with Binance
```python
# In rsi_price5.py:
DATA_SOURCE = "ccxt"
EXCHANGE_NAME = "binance"
DEFAULT_SYMBOL = "BTC/USDT"

# Then run:
python rsi_price5.py
```

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install CCXT separately:
```bash
pip install ccxt>=4.0.0
```

## Error Handling

The implementation includes comprehensive error handling:

1. **Missing CCXT library**: Clear error message with installation instructions
2. **Invalid exchange**: Lists available exchanges
3. **Invalid symbol**: Reports the error from the exchange
4. **Network issues**: Reports connection errors
5. **Invalid DATA_SOURCE**: Lists valid options

## Backward Compatibility

- Default configuration uses yfinance
- Existing users see no change in behavior
- CCXT is optional - application works without it if DATA_SOURCE is "yfinance"
- All existing yfinance functionality remains unchanged

## Implementation Details

### New Methods

1. **convert_symbol_to_ccxt(symbol)**: Converts yfinance format to CCXT format
2. **convert_timeframe_to_ccxt(interval)**: Converts timeframe if needed
3. **fetch_data_ccxt(exchange_name)**: Fetches OHLCV data using CCXT

### Updated Methods

1. **fetch_data()**: Routes to appropriate data source based on DATA_SOURCE

### Data Structure

Both data sources return a pandas DataFrame with:
- DatetimeIndex
- Columns: Open, High, Low, Close, Volume, RSI

This ensures compatibility across both data sources.

## Testing

All features have been tested:
- Symbol conversion
- RSI computation
- Data source routing
- Error handling
- DataFrame structure compatibility

## Notes

- CCXT provides access to real exchange data
- Different exchanges may have different symbol formats
- Some exchanges require API keys for certain features (not needed for public OHLCV data)
- Rate limits vary by exchange
