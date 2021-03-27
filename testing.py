import twitter
import sys

from twitter_api import *
from config import db_name

import os

# # Select no query terms
# q = 'lang:ar'

# sys.stderr.write('Filtering the public timeline for track="{}"'.format(q))

# twitter_api = oauth_login()
# twitter_stream = twitter.TwitterStream(auth=twitter_api.auth)

# stream = twitter_stream.statuses.filter(track=q)

# for tweet in stream:
#     print(tweet['text'])

twitter_api = oauth_login()
users = collect_users(twitter_api, screen_names=', '.join(['msha3bo', 'SafwatHatem', 'YahiaRag']))

user = users[0]
robust_collect_friends = partial(
        make_twitter_request, twitter_api.friends.list)

print('Collecting tweets')
tweets = collect_tweets(twitter_api, user, 200)

try:
    os.mkdir('user_tweets')
except OSError as error:
    print(error)

conn, c = create_db(db_name)

print('Storing tweets')
# store_tweets_json(tweets, user)
store_tweets(twitter_api, tweets, user, conn, c)


