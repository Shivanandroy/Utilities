# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:33:01 2019

@author: SH20018773
"""

import os
import tweepy as tw
import pandas as pd
import twitter_credentials
from datetime import datetime, timedelta
import re
from textblob import TextBlob


consumer_key= twitter_credentials.CONSUMER_KEY
consumer_secret= twitter_credentials.CONSUMER_SECRET
access_token= twitter_credentials.ACCESS_TOKEN
access_token_secret= twitter_credentials.ACCESS_TOKEN_SECRET

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

import re
def clean_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)"," ",tweet).split()).replace('amp;','')

def extract_tweets(keyword, lag_days=7, n_tweets=100):
# Define the search term and the date_since date as variables
    search_words = keyword + " -filter:retweets"
    date_since = datetime.now() - timedelta(days=lag_days)
    date_since = date_since.date()
    
    # Collect tweets
    tweets_cursor = tw.Cursor(api.search,  q=search_words,  lang="en", since=date_since, tweet_mode='extended').items(n_tweets)
    
    tweets=[]
    user_location = []
    dates = []
    for tweet_info in tweets_cursor:
        if "retweeted_status" in dir(tweet_info):
            tweet.append(tweet_info.retweeted_status.full_text)
        else:
            tweets.append(tweet_info.full_text)
            
        user_location.append(tweet_info.user.location)
        dates.append(tweet_info.created_at)
    
    tweets_df = pd.DataFrame({'tweet':tweets, 'date':dates,'location':user_location})
    tweets_df['date'] = pd.to_datetime(tweets_df['date'])
    tweets_df['clean_text'] = tweets_df.tweet.apply(clean_tweet)
    tweets_df['sentiment']=[TextBlob(line).sentiment.polarity for line in tweets_df.clean_text]
    return tweets_df



tweet_df = extract_tweets('pfizer',7,500)


s = tweet_df[['date','sentiment']].set_index('date').resample('30T').mean()








