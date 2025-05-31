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
        """Analyze news sentiment for a given stock"""
        try:
            # Get news articles
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Search for news
            news = self.news_api.get_everything(
                q=f"{symbol} OR {company_name}",
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            # Analyze sentiment
            sentiments = []
            for article in news['articles']:
                # Combine title and description for analysis
                text = f"{article['title']} {article['description']}"
                sentiment = TextBlob(text).sentiment
                
                sentiments.append({
                    'date': article['publishedAt'],
                    'title': article['title'],
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(sentiments)
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate daily sentiment
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
        """Analyze social media sentiment (placeholder for future implementation)"""
        # This is a placeholder for future implementation
        # Could integrate with Twitter API, Reddit API, etc.
        return {
            'twitter_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'overall_social_sentiment': 0.0
        }

    def get_combined_sentiment(self, symbol, company_name):
        """Combine news and social media sentiment"""
        news_sentiment = self.get_news_sentiment(symbol, company_name)
        social_sentiment = self.get_social_sentiment(symbol)
        
        if news_sentiment:
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