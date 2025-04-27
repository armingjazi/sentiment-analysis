# Trading Sentiment Analysis

This project uses financial data and sentiment analysis to predict stock price movements.  
It combines stock price information, sentiment data, and machine learning techniques to build predictive models.

## Features

- Download and preprocess stock and sentiment data
- Normalize, label, and clean financial time series
- Extract features and train predictive models
- Modular structure for data processing, model building, and evaluation
- Cache data for faster experiments
- Efficient batching mechanism for scalable data enrichment
- Intelligent retry mechanism for API rate limits
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

| Command                             | Description                                           |
| ----------------------------------- | ----------------------------------------------------- |
| `poetry run label <starting_batch>` | Label stock price data starting from a specific batch |
| `poetry run clean`                  | Clean and normalize datasets                          |
| `poetry run train`                  | Train the prediction model                            |

Alternatively, you can run specific scripts manually:

```bash
poetry run python trading_sentiment_analysis/main.py
```

### Step-by-Step Scripts

1. **Download Data**

   - Run `download_sentiment_data.py` to fetch sentiment datasets.
   - Use `twelve_data.py` to fetch stock price data.

2. **Label News Data in Batches**

   - News articles are labeled in **batches of 1000** entries.
   - Start labeling using:
     ```bash
     poetry run label <starting_batch>
     ```
   - Example:
     ```bash
     poetry run label 0
     ```
   - Each batch will be processed and saved separately.

3. **Clean and Normalize**

   - Run `poetry run clean` to preprocess and prepare datasets for modeling.

4. **Train the Model**
   - Run `poetry run train` to train the machine learning model based on processed features.

## Batching and Caching

- **Batching Mechanism:**

  - News articles are processed in batches of 1000 items to efficiently manage memory and processing time.
  - Each batch is saved individually to allow resuming from any batch index in case of interruptions.

- **Caching System:**

  - Stock price data retrieved from external APIs (like Twelve Data) is **cached locally**.
  - Cached files are reused in future runs to avoid redundant API requests, improving performance and respecting API rate limits.
  - The cache system automatically saves newly downloaded stock data for future use.

- **Rate Limiting Handling:**
  - Automatic retry mechanism is included to handle API rate limits, with smart long waits if necessary.

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
