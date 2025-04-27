# Trading Sentiment Analysis

This project uses financial data and sentiment analysis to predict stock price movements.  
It combines stock price information, sentiment data, and machine learning techniques to build predictive models.

## Features

- Download and preprocess stock and sentiment data
- Normalize, label, and clean financial time series
- Extract features and train predictive models
- Modular structure for data processing, model building, and evaluation
- Cache data for faster experiments
- Testing modules for reliability

## Data Sources

- **Kaggle**: Sentiment datasets
- **Twelve Data**: Stock price data

**Note:** Data from Kaggle and Twelve Data is **not included** in this repository. You will need your own API keys or access credentials to download the data.


## Installation

Make sure you have Python 3.12 installed.  
Then install the dependencies using Poetry:

```bash
poetry install
```

## Usage

You can run the main tasks via Poetry scripts:

| Command  | Description |
|----------|-------------|
| `poetry run label` | Label stock price data |
| `poetry run clean` | Clean and normalize datasets |
| `poetry run train` | Train the prediction model |

Alternatively, you can run specific scripts manually:

```bash
poetry run python trading_sentiment_analysis/main.py
```

### Step-by-Step Scripts

1. **Download Data**
    - Run `download_sentiment_data.py` to fetch sentiment datasets.
    - Use `twelve_data.py` to fetch stock price data.

2. **Label Data**
    - Run `poetry run label` or use `stock_price_label.py` manually to generate price movement labels.

3. **Clean and Normalize**
    - Run `poetry run clean` to preprocess, normalize, and clean the datasets.

4. **Train the Model**
    - Run `poetry run train` to train the machine learning model based on processed features.

## Configuration

Make sure to create a `.env` file if you need API keys (e.g., for Twelve Data or other stock APIs).  
You can also adapt settings like tickers, date ranges, or data sources directly in the scripts.

## Requirements

The main libraries used:

- `yfinance`
- `pandas`
- `numpy`
- `nltk`
- `matplotlib`
- `kagglehub`
- `python-dotenv`

(See `pyproject.toml` for complete versions.)

## Authors

- Armin Jazi 

