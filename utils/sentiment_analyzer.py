from textblob import TextBlob
from newsapi import NewsApiClient
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.news_api = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        self.logger = logging.getLogger(__name__)

    def get_news_sentiment(self, symbol, company_name, days=7):
        """Check what the news is saying about this stock lately"""
        try:
            # Figure out our date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Search for recent news articles
            news = self.news_api.get_everything(
                q=f"{symbol} OR {company_name}",
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            # Analyze the sentiment of each article
            sentiments = []
            for article in news['articles']:
                # Look at both title and description for better analysis
                text = f"{article['title']} {article['description']}"
                sentiment = TextBlob(text).sentiment
                
                sentiments.append({
                    'date': article['publishedAt'],
                    'title': article['title'],
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity
                })
            
            # Put it all in a nice DataFrame
            df = pd.DataFrame(sentiments)
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate average sentiment for each day
            daily_sentiment = df.groupby(df['date'].dt.date).agg({
                'polarity': 'mean',
                'subjectivity': 'mean'
            }).reset_index()
            
            return {
                'daily_sentiment': daily_sentiment.to_dict('records'),
                'overall_sentiment': {
                    'polarity': df['polarity'].mean(),
                    'subjectivity': df['subjectivity'].mean()
                },
                'article_count': len(sentiments)
            }
            
        except Exception as e:
            self.logger.error(f"Error in news sentiment analysis: {str(e)}")
            return None

    def get_social_sentiment(self, symbol):
        """Check social media vibes (coming soon!)"""
        # This is where we'd plug in Twitter, Reddit, etc.
        # For now, just returning neutral sentiment
        return {
            'twitter_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'overall_social_sentiment': 0.0
        }

    def get_combined_sentiment(self, symbol, company_name):
        """Combine news and social media sentiment into one score"""
        news_sentiment = self.get_news_sentiment(symbol, company_name)
        social_sentiment = self.get_social_sentiment(symbol)
        
        if news_sentiment:
            # Weight news more heavily since social media can be noisy
            overall_sentiment = (
                news_sentiment['overall_sentiment']['polarity'] * 0.7 +
                social_sentiment['overall_social_sentiment'] * 0.3
            )
        else:
            overall_sentiment = social_sentiment['overall_social_sentiment']
        
        return {
            'overall_sentiment': overall_sentiment,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment
        } 