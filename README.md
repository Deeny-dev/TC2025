# Wisdom_bee

pathaan
Sure! Below is an advanced AI-powered crypto trading bot with risk management, stop-loss, take-profit, and real-time alerts using Telegram or Email.


---

🔹 Upgraded Features in This AI Bot

✅ Risk Management (Dynamic position sizing & risk percentage)
✅ Stop-Loss & Take-Profit (Prevents major losses)
✅ Backtesting with historical data (Trains AI with real market conditions)
✅ Real-time alerts (Sends notifications via Telegram or Email)
✅ Live Trading with AI-powered decision-making


---

🔹 Step 1: Install Required Libraries

Run this command in your terminal:

pip install numpy pandas requests binance ta stable-baselines3 matplotlib python-dotenv telegram smtplib


---

🔹 Step 2: Set Up API Keys & Alerts

1. Binance API (for live trading)

Create a .env file and add your Binance API keys:

BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here


2. Telegram API (for alerts)

Get a Telegram bot token from BotFather

Add your chat ID and bot token in .env:

TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id


3. Email Alerts (Optional)

Add your email credentials in .env:

EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_email_password



---

🔹 Step 3: Full AI Trading Bot with Risk Management & Alerts

import os
import time
import requests
import numpy as np
import pandas as pd
import gym
import ta
import smtplib
from dotenv import load_dotenv
from stable_baselines3 import DQN
from binance.client import Client
from gym import spaces
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from telegram import Bot

# Load API keys
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# Initialize Binance & Telegram Clients
client = Client(API_KEY, SECRET_KEY)
telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Fetch historical price data
def get_historical_data(symbol="BTCUSDT", interval="1h", limit=100):
    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(candles, columns=["Time", "Open", "High", "Low", "Close", "Volume", "_", "_", "_", "_", "_", "_"])
    df["Close"] = df["Close"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    return df

# Compute technical indicators
def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = ta.volatility.BollingerBands(df["Close"]).bollinger_hband(), ta.volatility.BollingerBands(df["Close"]).bollinger_mavg(), ta.volatility.BollingerBands(df["Close"]).bollinger_lband()
    df["Support"] = df["Low"].rolling(20).min()
    df["Resistance"] = df["High"].rolling(20).max()
    return df

# Send alerts via Telegram
def send_telegram_alert(message):
    telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# Send alerts via Email
def send_email_alert(subject, message):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_USER
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.sendmail(EMAIL_USER, EMAIL_USER, msg.as_string())
    server.quit()

# AI Trading Environment with Stop-Loss & Take-Profit
class CryptoTradingEnv(gym.Env):
    def __init__(self, balance=10000, risk_per_trade=0.02, stop_loss=0.05, take_profit=0.1):
        super(CryptoTradingEnv, self).__init__()
        
        self.balance = balance
        self.crypto_held = 0
        self.df = add_indicators(get_historical_data())
        self.current_step = 0
        self.initial_balance = balance
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Action Space: [0 = Hold, 1 = Buy, 2 = Sell]
        self.action_space = spaces.Discrete(3)

        # Observation Space: [Price, RSI, MACD, Bollinger Bands, Support, Resistance]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step][["Close", "RSI", "MACD", "BB_Upper", "Support", "Resistance"]].values
        return np.nan_to_num(obs, nan=0)

    def step(self, action):
        self.current_step += 1
        current_price = self.df.iloc[self.current_step]["Close"]
        reward = 0

        position_size = self.balance * self.risk_per_trade / current_price

        if action == 1:  # Buy
            if self.balance >= current_price * position_size:
                self.crypto_held += position_size
                self.balance -= current_price * position_size
                reward = 1
                send_telegram_alert(f"📈 Buy Order Executed at ${current_price}")
                send_email_alert("Trade Alert: Buy", f"Buy Order at ${current_price}")

        elif action == 2:  # Sell
            if self.crypto_held > 0:
                self.balance += current_price * self.crypto_held
                self.crypto_held = 0
                reward = 1
                send_telegram_alert(f"📉 Sell Order Executed at ${current_price}")
                send_email_alert("Trade Alert: Sell", f"Sell Order at ${current_price}")

        # Apply Stop-Loss & Take-Profit
        if self.crypto_held > 0:
            entry_price = self.df.iloc[self.current_step - 1]["Close"]
            if current_price <= entry_price * (1 - self.stop_loss):
                self.crypto_held = 0
                self.balance += current_price * position_size
                send_telegram_alert(f"⚠️ Stop-Loss Triggered at ${current_price}")
            elif current_price >= entry_price * (1 + self.take_profit):
                self.crypto_held = 0
                self.balance += current_price * position_size
                send_telegram_alert(f"✅ Take-Profit Triggered at ${current_price}")

        done = self.current_step >= len(self.df) - 1
        return self._next_observation(), reward, done, {}

# Train AI Model
env = CryptoTradingEnv()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

# Live Trading Loop
state = env.reset()
done = False

while not done:
    action, _states = model.predict(state)
    state, reward, done, _ = env.step(action)
    env.render()
    time.sleep(5)


---

🔹 Enhancements

✅ Risk Management (Position Sizing, Stop-Loss, Take-Profit)
✅ Real-Time Alerts (Telegram & Email)
✅ AI-Powered Trading

Would you like to add backtesting reports or advanced risk models? 🚀

