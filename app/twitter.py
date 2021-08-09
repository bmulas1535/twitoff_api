import numpy as np
import os
import tweepy


def create_api():
  consumer_key = os.environ.get("TWITTER_API_KEY")
  consumer_secret = os.environ.get("TWITTER_API_SECRET")
  access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
  access_secret = os.environ.get("TWITTER_ACCESS_SECRET")

  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)
  api = tweepy.API(auth)

  return api

def get_tweets(name_1, name_2, api):
  tweet_list_1 = list()
  for tweet in tweepy.Cursor(api.user_timeline, id=name_1, full_text=True).items(200):
    tweet_list_1.append(tweet)

  tweet_list_2 = list()
  for tweet in tweepy.Cursor(api.user_timeline, id=name_2, full_text=True).items(200):
    tweet_list_2.append(tweet)

  y_t = [0] * len(tweet_list_1)
  y_t2 = [1] * len(tweet_list_2)

  y = np.asarray(y_t + y_t2).reshape(-1, 1)
  X = np.asarray(tweet_list_1 + tweet_list_2)

  return X, y

