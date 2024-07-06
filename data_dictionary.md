# Data Dictionary: Pharmaceutical Sector Performance Prediction Model

**Last Updated:** [Current Date]

## 1. Time-related Variables

| Variable Name | Description | Data Type | Format | Frequency | Source |
|---------------|-------------|-----------|--------|-----------|--------|
| Date | The date of the observation | Date | YYYY-MM-DD | Daily | N/A (Index) |

## 2. Target Variable

| Variable Name | Description | Data Type | Unit | Calculation | Source |
|---------------|-------------|-----------|------|-------------|--------|
| Pharma_Index | The daily return of the equal-weighted pharmaceutical index | Float | Percentage | (Today's Index Value / Yesterday's Index Value) - 1 | Calculated from stock prices |

## 3. Stock-related Variables

| Variable Name | Description | Data Type | Unit | Calculation | Source |
|---------------|-------------|-----------|------|-------------|--------|
| PFE, JNJ, MRK, ABBV, BMY | Daily returns of individual pharmaceutical stocks (Pfizer, Johnson & Johnson, Merck, AbbVie, Bristol-Myers Squibb) | Float | Percentage | (Today's Closing Price / Yesterday's Closing Price) - 1 | Yahoo Finance |

## 4. Technical Indicators

| Variable Name | Description | Data Type | Unit | Calculation | Source |
|---------------|-------------|-----------|------|-------------|--------|
| Pharma_Index_Lag_1, Pharma_Index_Lag_5, Pharma_Index_Lag_20 | Lagged values of the Pharma_Index (1-day, 5-day, and 20-day lags) | Float | Percentage | Pharma_Index shifted by respective number of days | Derived from Pharma_Index |
| Pharma_Index_MA_50, Pharma_Index_MA_200 | 50-day and 200-day moving averages of the Pharma_Index | Float | Percentage | Average of Pharma_Index over the specified number of trading days | Derived from Pharma_Index |
| RSI | Relative Strength Index of the Pharma_Index | Float | Numeric (0-100) | 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss over 14 days | Derived from Pharma_Index |

## 5. Economic Indicators

| Variable Name | Description | Data Type | Unit | Frequency | Source |
|---------------|-------------|-----------|------|-----------|--------|
| GDP_Growth | Quarterly GDP growth rate | Float | Percentage | Quarterly (interpolated to daily) | U.S. Bureau of Economic Analysis |
| Inflation_Rate | Monthly inflation rate | Float | Percentage | Monthly (interpolated to daily) | U.S. Bureau of Labor Statistics |
| Interest_Rate | Federal Funds Rate | Float | Percentage | Updated as announced by the Federal Reserve (interpolated to daily) | Federal Reserve Economic Data (FRED) |

## 6. Healthcare-specific Indicators

| Variable Name | Description | Data Type | Unit | Frequency | Source |
|---------------|-------------|-----------|------|-----------|--------|
| Healthcare_Spending | Total healthcare spending in the U.S. | Float | Billions of USD | Annual (interpolated to daily) | Centers for Medicare & Medicaid Services |
| FDA_Approvals | Number of FDA drug approvals | Integer | Count | Daily (cumulative for the year) | U.S. Food and Drug Administration |

## 7. Market Sentiment Indicators

| Variable Name | Description | Data Type | Unit | Frequency | Source |
|---------------|-------------|-----------|------|-----------|--------|
| VIX | CBOE Volatility Index, representing market expectation of near-term volatility | Float | Index Points | Daily | Chicago Board Options Exchange |
| Pharma_News_Sentiment | Sentiment score of pharmaceutical industry news | Float | Score (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive) | Daily | News API and sentiment analysis algorithm |

## 8. Derived Features

| Variable Name | Description | Data Type | Unit | Calculation | Source |
|---------------|-------------|-----------|------|-------------|--------|
| Pharma_Index_Volatility | 30-day rolling standard deviation of Pharma_Index returns | Float | Percentage | Standard deviation of Pharma_Index over the past 30 trading days | Derived from Pharma_Index |
| Pharma_Index_Momentum | 14-day momentum of the Pharma_Index | Float | Percentage | (Today's Pharma_Index / Pharma_Index 14 days ago) - 1 | Derived from Pharma_Index |

## Notes:
1. All daily data is based on trading days, excluding weekends and stock market holidays.
2. Economic indicators with lower frequency (monthly, quarterly, annual) are interpolated to daily values for modeling purposes.
3. Missing values are handled through forward filling for most variables, except for derived features which may use specific imputation methods.

This Data Dictionary provides a comprehensive overview of the variables used in your project. It includes:

- A clear description of each variable
- The data type and unit of measurement
- The frequency of data collection or update
- The source of the data
- Any calculations or derivations used to create the variable

When creating your own Data Dictionary:

- Ensure all variables used in your analysis are included
- Be specific about data sources, especially for external data
- Clearly explain any derived or calculated variables
- Include information about how missing data is handled
- Update the dictionary as new variables are added or existing ones are modified

This document will be invaluable for anyone working on or reviewing your project, ensuring clarity and consistency in the understanding and use of your data variables.
