import time

mongo_db = 'twitter_stream'
collected_users_list = []
tweets_count = 0
tweets_per_user = 200
friends_per_user = 50
limit_depth = 2  # The depth of the tree of users to collect tweets from
seed_names = ', '.join(['YahiaRag', 'MSha3bo', 'SafwatHatem'])
start_time = time.time()