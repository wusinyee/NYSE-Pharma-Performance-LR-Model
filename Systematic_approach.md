# Comprehensive Guide to ML Project on Pharmaceutical Performance Linear Regression Model

## 1. Data Collection and Preprocessing

**Explanation:** Gather data from various sources and prepare it for analysis.

**Example:** Collecting stock data for major pharmaceutical companies and relevant economic indicators.

```python
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# Define pharmaceutical companies and date range
pharma_tickers = ['PFE', 'JNJ', 'MRK', 'ABBV', 'BMY']
start_date = '2010-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch stock data
def get_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data['Adj Close']

stock_data = get_stock_data(pharma_tickers, start_date, end_date)

# Create pharmaceutical index (equal-weighted for simplicity)
stock_data['Pharma_Index'] = stock_data.mean(axis=1)

# Calculate returns
returns_data = stock_data.pct_change()

# Fetch economic data (example with GDP growth - you'd need to source this data)
gdp_growth = pd.read_csv('gdp_growth.csv', parse_dates=['Date'], index_col='Date')

# Merge datasets
df = pd.concat([returns_data, gdp_growth], axis=1).dropna()

print(df.head())
```

## 2. Exploratory Data Analysis (EDA)

**Explanation**: Analyze the data to understand patterns, trends, and relationships.

**Example**: Visualizing the pharmaceutical index performance and its correlation with economic indicators.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot pharmaceutical index performance
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Pharma_Index'].cumsum(), label='Pharma Index')
plt.title('Cumulative Returns of Pharmaceutical Index')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of returns
plt.figure(figsize=(10, 6))
sns.histplot(df['Pharma_Index'], kde=True)
plt.title('Distribution of Pharmaceutical Index Returns')
plt.xlabel('Returns')
plt.show()
```

## 3. Feature Engineering

**Explanation**: Create new features that might help predict pharmaceutical sector performance.

**Example**: Creating lagged features and moving averages.

```python
# Create lagged features
for lag in [1, 5, 20]:
    df[f'Pharma_Index_Lag_{lag}'] = df['Pharma_Index'].shift(lag)

# Create moving averages
for window in [50, 200]:
    df[f'Pharma_Index_MA_{window}'] = df['Pharma_Index'].rolling(window=window).mean()

# Create relative strength index (RSI)
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['Pharma_Index'])

# Drop NaN values created by lagged features
df.dropna(inplace=True)

print(df.head())
```
