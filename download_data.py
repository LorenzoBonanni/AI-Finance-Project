


import yfinance as yf
import re

if __name__ == "__main__":
    # real_data extraction

    # Apple Inc. (AAPL) - Technology
    # Alphabet Inc. (GOOGL) - Technology
    # Amazon.com Inc. (AMZN) - Retail
    # Microsoft Corporation (MSFT) - Technology
    # Johnson & Johnson (JNJ) - Healthcare
    # Tesla (TSLA) - Consumer Cyclical
    # Meta Platforms Inc. (META) - Technology
    # Procter & Gamble Co. (PG) - Consumer Goods
    # Visa Inc. (V) - Financial Services
    # JPMorgan Chase & Co. (JPM) - Financial Services
    # Exxon Mobil Corporation (XOM) - Energy
    # Coca-Cola Co. (KO) - Beverages
    # Walmart Inc. (WMT) - Retail
    # Pfizer Inc. (PFE) - Healthcare
    # Home Depot Inc. (HD) - Home Improvement
    # Mastercard Incorporated (MA) - Financial Services
    # Nike Inc. (NKE) - Consumer Goods
    # Walt Disney Co. (DIS) - Entertainment
    # McDonald’s Corporation (MCD) - Restaurants
    # UnitedHealth Group Incorporated (UNH) - Healthcare

    stocks = ['AAPL','GOOGL','AMZN','MSFT','AAPL','JNJ','TSLA','META','PG', 'V',
              'JPM','XOM','KO','WMT','PFE', 'HD','MA','NKE','DIS','MCD', 'UNH']
    start_date = '2013-01-01'
    end_date = '2023-05-18'

    for stock in stocks:
        data = yf.download(stock, start=start_date, end=end_date)
        data.to_csv(f'./data_feed/{stock}.csv')
