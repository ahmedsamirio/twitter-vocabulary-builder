import time

collected_users_list = []
tweets_count = 0
tweets_per_user = 3
friends_per_user = 2
limit_depth = 3  # The depth of the tree of users to collect tweets from
seed_name = "Wadoudagain"
db_name = 'twitter_stream.db'
start_time = time.time()