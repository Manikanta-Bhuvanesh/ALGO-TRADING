import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product
from numba import njit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

default_directory = r'C:\Users\Bhuvanesh\Desktop\SCANNER_BRUTE_SUPER_TREND'
os.chdir(default_directory)

def DataFetcher(symbol, interval='1d'):
    suffixes = ['.NS', '.BO']
    for suffix in suffixes:
        try:
            data = yf.download(symbol + suffix, interval=interval, progress=False)
            data.drop(columns='Adj Close', inplace=True)
            data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            data['Time'] = data.index
            return data, symbol + suffix
        except Exception as e:
            continue
    return pd.DataFrame(), None

@njit
def calculate_supertrend_fast(high, low, close, volume, period, multiplier):
    atr = np.zeros_like(close)
    supertrend = np.zeros_like(close)
    direction = np.zeros_like(close)
    
    # Calculate ATR
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    # Calculate volume factor
    volume_ma = np.zeros_like(volume)
    volume_ma[period-1] = np.mean(volume[:period])
    for i in range(period, len(volume)):
        volume_ma[i] = (volume_ma[i-1] * (period - 1) + volume[i]) / period
    volume_factor = volume / volume_ma
    
    # Calculate SuperTrend
    hl2 = (high + low) / 2
    basic_upperband = hl2 + (multiplier * atr * volume_factor)
    basic_lowerband = hl2 - (multiplier * atr * volume_factor)
    
    for i in range(period, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(basic_lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(basic_upperband[i], supertrend[i-1])
        
        if close[i] > supertrend[i]:
            direction[i] = 1
        else:
            direction[i] = -1

    return supertrend, direction

def calculate_super_trend(df, multiplier, period):
    supertrend, direction = calculate_supertrend_fast(
        df['high'].values, df['low'].values, df['close'].values, 
        df['volume'].values, period, multiplier
    )
    
    df['super_trend'] = supertrend
    df['signal'] = direction
    df['Position'] = df['signal'].diff()
    df['Buy/Sell'] = df['Position'].map({2: 'Buy', -2: 'Sell', 0: 'Hold'})
    df['VSMA'] = df['volume'].rolling(window=period).mean()
    
    return df

def calculate_super_trend_performance(df):
    if 'signal' not in df.columns:
        raise ValueError("Signal column not created. Check calculate_super_trend function.")

    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['strategy'] = df['signal'].shift(1) * df['returns']
    df['creturns'] = df['returns'].cumsum().apply(np.exp)
    df['cstrategy'] = df['strategy'].cumsum().apply(np.exp)
    perf = df['cstrategy'].iloc[-1]
    buy_and_hold = df['creturns'].iloc[-1]
    outperf = perf - buy_and_hold
    return round(perf, 6), round(buy_and_hold, 6), round(outperf, 6)

def optimize_super_trend_parameters(df, multipliers, periods):
    best_performance = -np.inf
    best_params = None
    for multiplier, period in product(multipliers, periods):
        temp_df = df.copy()
        temp_df = calculate_super_trend(temp_df, multiplier, period)
        performance = calculate_super_trend_performance(temp_df)[0]
        if performance > best_performance:
            best_performance = performance
            best_params = (multiplier, period)
    return best_params

def calculate_profit_percentage(df):
    initial_principal = 100000  # Start with $100,000
    cash = initial_principal
    shares = 0

    for index, row in df.iterrows():
        if row['Buy/Sell'] == 'Buy' and cash > 0:
            shares_bought = cash // row['close']
            shares += shares_bought
            cash -= shares_bought * row['close']
        elif row['Buy/Sell'] == 'Sell' and shares > 0:
            cash += shares * row['close']
            shares = 0

    # If there are any shares left, sell them at the last recorded price
    if shares > 0:
        cash += shares * df['close'].iloc[-1]
        shares = 0

    final_principal = cash
    profit = final_principal - initial_principal
    profit_percentage = (profit / initial_principal) * 100
    return profit_percentage

def calculate_trade_percentages(df):
    if 'Buy/Sell' not in df.columns:
        raise ValueError("Buy/Sell column not created. Check calculate_super_trend function.")
    
    buy_prices = []
    sell_prices = []

    latest_close = df.tail(1)['close'].values[0]

    for index, row in df.iterrows():
        if row['Buy/Sell'] == 'Buy':
            buy_prices.append(row['close'])
        elif row['Buy/Sell'] == 'Sell' and buy_prices:
            sell_prices.append(row['close'])

    if len(buy_prices) > len(sell_prices):
        sell_prices.append(latest_close)

    trade_percentages = [(sell - buy) / buy * 100 for buy, sell in zip(buy_prices, sell_prices)]

    success_percentages = [p for p in trade_percentages if p > 0]
    loss_percentages = [p for p in trade_percentages if p < 0]

    average_success_percentage = np.mean(success_percentages) if success_percentages else 0
    average_loss_percentage = np.mean(loss_percentages) if loss_percentages else 0

    return average_success_percentage, average_loss_percentage

def get_market_cap(symbol):
    def fetch_market_cap(ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            market_cap = info.get('marketCap')
            return market_cap
        except Exception:
            return None

    market_cap = fetch_market_cap(symbol + '.NS')

    if market_cap is None:
        market_cap = fetch_market_cap(symbol + '.BO')

    if market_cap is None:
        return -1

    market_cap_in_crores = market_cap / 10**7
    return market_cap_in_crores

def process_stock(stock):
    multipliers = np.arange(1.5, 3.5, 0.5)
    periods = np.arange(7, 22, 1)
    try:
        df, symbol = DataFetcher(stock)
        if df.empty:
            return {'symbol': stock, 'error': 'Data fetch failed'}

        best_params = optimize_super_trend_parameters(df.copy(), multipliers, periods)
        temp_df = calculate_super_trend(df, best_params[0], best_params[1])
        profit_percentage = calculate_profit_percentage(temp_df)
        average_trade_percentage = calculate_trade_percentages(temp_df)
        market_cap = get_market_cap(stock)
        return {'symbol': symbol, 'multiplier': best_params[0], 'period': best_params[1],
                'average_success_percentage': average_trade_percentage[0],
                'average_loss_percentage': average_trade_percentage[1],
                'percentage': profit_percentage, 'Market Cap cr': market_cap}
    except Exception as e:
        return {'symbol': stock, 'error': str(e)}

if __name__ == '__main__':
    # Read the stock symbols
    stocks_df = pd.read_csv('STOCKS_1.csv')
    stocks_df = stocks_df.dropna(subset=['Symbol'])
    stocks = stocks_df['Symbol']

    

    all_stocks_data = []
    failed_stocks = []
    success_stocks = []

    # Use multiprocessing.Pool to parallelize the process
    with Pool(processes=cpu_count()-1) as pool:
        results = list(tqdm(pool.imap(process_stock, stocks), total=len(stocks), desc="Processing stocks"))
    
    for result in results:
        if 'error' in result:
            failed_stocks.append(result['symbol'])
        else:
            all_stocks_data.append(result)
            success_stocks.append(result['symbol'])

    all_stocks_df = pd.DataFrame(all_stocks_data)
    csv_file_path = 'Paramns_stocks.csv'
    all_stocks_df.to_csv(csv_file_path, index=False)

    print(f"Processing completed. Successful stocks: {len(success_stocks)}, Failed stocks: {len(failed_stocks)}")
    print(f"Results saved to {csv_file_path}")
