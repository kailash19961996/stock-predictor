import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Stock Prediction App")

# User input for stock selection
stock = st.text_input("Enter the stock ticker (e.g., AAPL, GOOG):")

# User input for date range
start_date = st.date_input("Start date", datetime(2018, 1, 1))
end_date = st.date_input("End date", datetime.now())

# Fetch and display stock data
if stock:
    data = yf.download(stock, start=start_date, end=end_date)
    st.write(f"Stock data for {stock} from {start_date} to {end_date}")
    st.line_chart(data['Close'])

    # Fetch news
    newsapi = NewsApiClient(api_key='75fd86230a9d4e05b9c8ca9beec27d03')

    def fetch_news(stock, start_date, end_date):
        all_articles = newsapi.get_everything(
            q=stock,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
        )
        return all_articles

    articles = fetch_news(stock, start_date, end_date)
    st.subheader(f"The top 5 atricles are :")
    i = 0
    for article in articles['articles']:
        st.write(f"**{article['title']}**")
        st.write(f"*{article['publishedAt']}*")
        st.write(f"{article['description']}")
        st.write("---")
        i += 1
        if i > 4:
            break

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(articles):
        sentiments = []
        dates = []
        for article in articles['articles']:
            if article['description']:  # Check if the description is not None
                sentiment = analyzer.polarity_scores(article['description'])
                sentiments.append(sentiment)
                dates.append(pd.to_datetime(article['publishedAt']).date())
        return sentiments, dates

    sentiments, sentiment_dates = analyze_sentiment(articles)
    st.write(sentiments[:5])

    # Predictive model
    def prepare_data(sentiments, sentiment_dates, stock_data):
        sentiment_scores = [s['compound'] for s in sentiments]
        aligned_sentiments = []
        stock_changes = []

        for date, score in zip(sentiment_dates, sentiment_scores):
            if date in stock_data.index:
                aligned_sentiments.append(score)
                stock_changes.append(stock_data.loc[date]['Close'])

        stock_changes_pct = pd.Series(stock_changes).pct_change().dropna().values
        return np.array(aligned_sentiments[:-1]).reshape(-1, 1), stock_changes_pct

    aligned_sentiments, stock_changes = prepare_data(sentiments, sentiment_dates, data)
    
    # Debug print to check lengths of aligned data
    st.write(f"Number of aligned sentiment scores: {len(aligned_sentiments)}")
    st.write(f"Number of stock changes: {len(stock_changes)}")

    # Ensure there are enough samples for train-test split
    if len(aligned_sentiments) > 1 and len(stock_changes) > 1:
        X_train, X_test, y_train, y_test = train_test_split(aligned_sentiments, stock_changes, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.write(f"Model Score: {model.score(X_test, y_test)}")
    else:
        st.write("Not enough data to perform train-test split.")
