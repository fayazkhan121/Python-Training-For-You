from flask import Flask, request, jsonify
import yfinance as yf

app = Flask(__name__)

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None, str(e)

# API endpoint to get stock data
@app.route('/api/stock', methods=['GET'])
def get_stock():
    ticker = request.args.get('ticker', '').strip().upper()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()

    if not ticker or not start_date or not end_date:
        return jsonify({"error": "Please provide ticker, start_date, and end_date."}), 400

    try:
        stock_data, error = fetch_stock_data(ticker, start_date, end_date)

        if error:
            return jsonify({"error": error}), 500

        if stock_data is not None and not stock_data.empty:
            # Convert data to JSON format
            stock_data_json = stock_data.to_dict(orient='records')
            return jsonify(stock_data_json)
        else:
            return jsonify({"error": "No data available for the given period."}), 404

    except ValueError as ve:
        return jsonify({"error": f"Date format error: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
