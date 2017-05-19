import numpy as np
import pandas as pd
import datetime as dt
import math
import time
import util
import sys


def get_indicators (symbols, start_date, end_date, lookback=14):

  # Construct an appropriate DatetimeIndex object.
  dates = pd.date_range(start_date, end_date)

  # Read all the relevant price data (plus SPY) into a DataFrame.
  price = util.get_data(symbols, dates)

  # Add SPY to the symbol list for convenience.
  symbols.append('SPY')


  #Simple Moving Average
  sma = price.rolling(window=lookback,min_periods=lookback).mean()

  smaRatio = sma.copy()
  smaRatio = price / sma


  # Exponentially Moving Average
  ewma = price.ewm(ignore_na=False, span=lookback, min_periods=0, adjust=True).mean()
  ewmaRatio = ewma.copy()
  ewmaRatio = price / ewma

  #Momentum
  momentum = price.copy()
  momentum /= momentum.shift(lookback)
  momentum -= 1
  momentum = momentum.fillna(0)


  #Bollinger Bands
  rolling_std = price.rolling(window=lookback,min_periods=lookback).std()
  top_band = sma + (2 * rolling_std)
  bottom_band = sma - (2 * rolling_std)

  bbp = (price - bottom_band) / (top_band - bottom_band)



  data = pd.concat([price, sma, smaRatio, bbp, ewma, ewmaRatio, momentum], keys=['Price', 'SMA', 'SMA Ratio', 'BBP', 'EMA', 'EMA Ratio', 'Momentum'], axis=1)
  return data

