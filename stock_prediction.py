import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

def get_stock_data(ticker, exchange, hist_days):
    """Fetch stock data from Yahoo Finance"""

    # Add exchange suffix
    if exchange.upper() == 'NSE':
        symbol = f"{ticker}.NS"
    else:
        symbol = f"{ticker}.BO"

    # Calculate start date based on user input
    end_date = datetime.now()
    start_date = end_date - timedelta(days=hist_days)

    print(f"\nFetching {hist_days} days of data for {symbol}...")
    stock = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if stock.empty:
        raise ValueError(f"No data found for {symbol}. Please check ticker.")

    return stock, symbol


def prepare_data(stock_data):
    prices = stock_data['Close'].dropna().values
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices
    return X, y

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def make_predictions(model, X, future_days=30):
    y_pred = model.predict(X)
    last_day = X[-1][0]
    future_X = np.arange(last_day + 1, last_day + future_days + 1).reshape(-1, 1)
    future_pred = model.predict(future_X)
    return y_pred, future_pred


def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def get_recommendation(current_price, predicted_price):
    price_change = ((predicted_price - current_price) / current_price) * 100
    if price_change > 5:
        return "BUY", price_change
    elif price_change < -5:
        return "SELL", price_change
    else:
        return "HOLD", price_change


def plot_results(X, y, y_pred, future_pred, symbol):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(X[-60:], y[-60:], label='Actual Price', linewidth=2)
    ax1.plot(X[-60:], y_pred[-60:], label='Model Prediction', linestyle='--', linewidth=2)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Price (₹)')
    ax1.set_title(f'{symbol} - Historical (Last 60 Days)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    future_days = np.arange(1, len(future_pred) + 1)
    ax2.plot(future_days, future_pred, label='Predicted Price', linewidth=2)
    ax2.set_xlabel('Days Ahead')
    ax2.set_ylabel('Price (₹)')
    ax2.set_title(f'{symbol} - Future Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("STOCK PRICE PREDICTOR - LINEAR REGRESSION MODEL")
    print("=" * 60)

    ticker = input("\nEnter ticker symbol (e.g., RELIANCE, TCS, INFY): ").strip().upper()
    exchange = input("Enter exchange (NSE/BSE): ").strip().upper()

    # Ask user how much data to fetch
    print("\nHow much historical data?")
    hist_value = int(input("Enter value (e.g. 1, 6, 365): "))
    hist_unit = input("Unit (D = Days, M = Months, Y = Years): ").strip().upper()

    if hist_unit == 'Y':
        hist_days = hist_value * 365
    elif hist_unit == 'M':
        hist_days = hist_value * 30
    else:
        hist_days = hist_value

    # Ask future prediction period
    future_days = int(input("\nHow many days ahead to predict? (e.g., 30): "))

    if exchange not in ['NSE', 'BSE']:
        print("Invalid exchange. Using NSE by default.")
        exchange = 'NSE'

    try:
        stock_data, symbol = get_stock_data(ticker, exchange, hist_days)
        X, y = prepare_data(stock_data)
        print("Training linear regression model...")
        model = train_model(X, y)

        y_pred, future_pred = make_predictions(model, X, future_days)

        r2 = calculate_r2(y, y_pred)
        current_price = float(y[-1])
        predicted_price = float(future_pred[-1])

        recommendation, price_change = get_recommendation(current_price, predicted_price)

        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"\nStock Symbol: {symbol}")
        print(f"Historical Data Used: {hist_days} days")
        print(f"Current Price: ₹{current_price:.2f}")
        print(f"Predicted Price ({future_days} days): ₹{predicted_price:.2f}")
        print(f"Expected Change: {price_change:+.2f}%")
        print(f"R² Score: {r2*100:.2f}%")

        confidence = "High" if r2>0.7 else "Moderate" if r2>0.4 else "Low"
        print(f"Model Confidence: {confidence}")
        print(f"\n{'*'*60}")
        print(f"RECOMMENDATION: {recommendation}")
        print(f"{'*'*60}")
        print("\nNote: Educational model. Not financial advice.")

        print("\nGenerating charts...")
        plot_results(X, y, y_pred, future_pred, symbol)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check ticker and try again.")

if __name__ == "__main__":
    main()