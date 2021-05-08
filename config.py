import time

mongo_db = 'twitter_stream_2'
collected_users_list = []
tweets_count = 0
tweets_per_user = 200
friends_per_user = 100
limit_depth = 0  # The depth of the tree of users to collect tweets from
seed_names = ', '.join(['YahiaRag', 'MSha3bo', 'SafwatHatem'])
start_time = time.time()

