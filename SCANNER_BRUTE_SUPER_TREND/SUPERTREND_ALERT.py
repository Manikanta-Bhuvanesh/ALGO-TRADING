import pandas as pd
import numpy as np
import yfinance as yf
from numba import njit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
import time

default_directory = r'C:\Users\Bhuvanesh\Desktop\SCANNER_BRUTE_SUPER_TREND'
os.chdir(default_directory)

def DataFetcher(symbol, interval='1d'):
    data = yf.download(symbol, interval=interval, progress=False)
    data.drop(columns='Adj Close', inplace=True)
    data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    data['Time'] = data.index
    return data

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

def process_symbol(row):
    try:
        temp_df = DataFetcher(row['symbol'])
        temp_df = calculate_super_trend(temp_df, row['multiplier'], row['period'])
        result = temp_df[['Time', 'close', 'Buy/Sell', 'VSMA']].tail(1).assign(
            average_success_percentage=row['average_success_percentage'],
            average_loss_percentage=row['average_loss_percentage'],
            Profit=row['percentage'],
            symbol=row['symbol'],
            Market_Cap_cr=row['Market Cap cr']
        )
        return result
    except Exception as e:
        print(f"Error processing {row['symbol']}: {str(e)}")
        return None

# Send email alert
def send_email_alert(all_stocks_df, new_stocks_df):
    if not all_stocks_df.empty:
        all_stocks_df = all_stocks_df.reset_index(drop=True)
        all_stocks_df.loc[:, 'Tag'] = 'Old'

    if not new_stocks_df.empty:
        new_stocks_df = new_stocks_df.reset_index(drop=True)
        new_stocks_df.loc[:, 'Tag'] = 'New'

    
    # Combine the new stocks with the existing ones
    combined_df = pd.concat([new_stocks_df,all_stocks_df])

    # Save combined_df as a CSV file
    csv_filename = "Stock_Alerts.csv"
    combined_df.to_csv(csv_filename, index=False)

    sender_email = "bhuvanesh.valiveti@gmail.com"
    receiver_emails = ["bhuvanesh.valiveti@gmail.com","guruteja26@gmail.com","manas.baggu.official@gmail.com","manishsikakolli@gmail.com","tpvsssaketh@gmail.com"]
    subject = "BRUTE Super Trend Stock Alerts"
    body = "Stock alerts have been generated. Please find the details in the attached CSV file."

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, "bcea clul fryl htqw")  # Replace with your actual password

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the CSV file to the email
    with open(csv_filename, "rb") as attachment:
        part = MIMEApplication(attachment.read(), _subtype="csv")
        part.add_header('Content-Disposition', 'attachment', filename=csv_filename)
        msg.attach(part)

    server.send_message(msg)
    server.quit()

# Main loop to run the strategy
def run_strategy():
    df = pd.read_csv('Paramns_stocks.csv')
    df=df[(df['average_success_percentage'] > 4) & (df['average_loss_percentage'] > -4) & (df['percentage'] >100)]
    previous_stocks_df = pd.DataFrame(columns=['symbol', 'Buy/Sell', 'close', 'Time', 'VSMA', 'average_success_percentage', 'average_loss_percentage', 'Profit', 'Market_Cap_cr'])
    while True:
        try:
                
            num_cores = cpu_count() - 1  # Leave one core free
        
            # Create a pool of worker processes
            with Pool(num_cores) as pool:
                # Use tqdm to show a progress bar
                results = list(tqdm(pool.imap(process_symbol, df.to_dict('records')), total=len(df), desc="Processing stocks"))

            all_stocks_df = pd.concat([r for r in results if r is not None])
            
            all_stocks_df = all_stocks_df[(all_stocks_df['Buy/Sell'] == 'Buy') & (all_stocks_df['VSMA'] > 30000) & (all_stocks_df['close']< 1700)]
            new_stocks_df = all_stocks_df[~all_stocks_df['symbol'].isin(previous_stocks_df['symbol'])]
            # old_stocks_df = previous_stocks_df[all_stocks_df['symbol'].isin(previous_stocks_df['symbol'])]

            if not new_stocks_df.empty:
                send_email_alert(previous_stocks_df, new_stocks_df)
                previous_stocks_df = all_stocks_df  # Update the previous stocks
                time.sleep(15*60)  # Wait for 1 minute before checking again
        except Exception as e:
            continue


if __name__ == '__main__':

    
    run_strategy()