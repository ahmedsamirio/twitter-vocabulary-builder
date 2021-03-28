import time

mongo_db = 'twitter_stream'
collected_users_list = []
tweets_count = 0
tweets_per_user = 200
friends_per_user = 20
limit_depth = 3  # The depth of the tree of users to collect tweets from
seed_names = ', '.join(['msha3bo', 'SafwatHatem', 'YahiaRag'])
start_time = time.time()