# Kills warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import warnings
warnings.filterwarnings("ignore")

# imports
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import track

console = Console()
console.clear()
console.rule(style="bold bright_cyan")
console.print("[bold bright_cyan] STOCK VISIONARY [/]", justify="center")
console.rule(style="bold bright_cyan")

#console.rule("Stock Visionary", style="bold cyan")


# User inputs
stock = Prompt.ask("[bold green]Enter stock[/bold green]")
per = Prompt.ask("[bold green]Enter period (1mo,3mo,6mo,1y,2y,5y,max)[/bold green]")
for step in track(range(100), description="Processing"):
    # Simulate work
    pass
print()

# Download data
data = yf.download(stock.upper(), period=per, auto_adjust=True, progress=False)
if data.empty:
    console.print("[bold red]No data found for this ticker![/bold red]")
    exit()

# Flatten multi-index columns if present
data.columns = [col[0] for col in data.columns]
data.to_csv("data.csv", index_label="Date")

# Load OHLCV features (much better than Close only)
pdata = pd.read_csv("data.csv")
features = pdata[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Scale all features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

# User inputs
prediction_days = int(Prompt.ask("[bold green]Enter lookback days (30-60)[/bold green]"))
forecast_horizon = int(Prompt.ask("[bold green]Enter future days to predict (20-30)[/bold green]"))
print()

# SAFETY CHECK
min_len = prediction_days + forecast_horizon
if len(scaled) <= min_len:
    print(f"Need >{min_len} rows. Got {len(scaled)}")
    exit()

# Train/test split (80/20)
split = int(0.8 * len(scaled))
train_scaled = scaled[:split]

console.print(f"[bold green]Training on {len(train_scaled)} days.[/bold green]")

# Create sequences using training data only
x, y = [], []
n_features = scaled.shape[1]  # 5 for OHLCV

for i in range(prediction_days, len(train_scaled) - forecast_horizon + 1):
    x.append(train_scaled[i - prediction_days:i, :])  # All 5 features
    y.append(train_scaled[i:i + forecast_horizon, 3])  # Predict Close only

x = np.array(x)
y = np.array(y)

console.print(f"[bold green]Created {len(x)} training sequences[/bold green]")
print()
# Enhanced stacked LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(prediction_days, n_features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(forecast_horizon)
])

model.compile(optimizer="adam", loss="mse")

# Early stopping
early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

# Train
model.fit(x, y, epochs=100, batch_size=8, verbose=1, callbacks=[early_stop])

# Predict future using LAST prediction_days from ALL data
last_sequence = scaled[-prediction_days:, :].reshape((1, prediction_days, n_features))
next_scaled = model.predict(last_sequence, verbose=0)

# Inverse transform - only Close prices (index 3)
dummy = np.zeros((len(next_scaled[0]), n_features))
dummy[:, 3] = next_scaled[0]  # Place predictions in Close column
next_prices = scaler.inverse_transform(dummy)[:, 3]

console.print(f"[bold blue]Stock predictions for: {stock.upper()}[/bold blue]")
last_close = pdata["Close"].iloc[-1]
for i, price in enumerate(next_prices, 1):
    change_pct = ((price - last_close) / last_close) * 100
    console.print(f"[bold cyan]Day {i:2d}:[/bold cyan][bold green] ${price:8.2f} ({change_pct:+6.1f}%)[/bold green]")
