import re
import emoji

def tweet_cleaning(tweet):
    tweet = tweet.replace('#', '')
    tweet = emoji.replace_emoji(tweet, '')
    tweet = re.sub("@[A-Za-z0-9]+","",tweet)
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
    tweet = tweet.replace('@', '')
    
    return tweet