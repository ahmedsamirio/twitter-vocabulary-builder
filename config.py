import time

collected_users_list = []
tweets_count = 0
tweets_per_user = 100
friends_per_user = 1
limit_depth = 1  # The depth of the tree of users to collect tweets from
seed_names = ', '.join(['msha3bo', 'SafwatHatem', 'YahiaRag'])
start_time = time.time()