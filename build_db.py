import os
import sqlite3
from twitter_api import *
from db_manipulation import *


if __name__ == "__main__":

    twitter_api = oauth_login()

    db_name = 'twitter_stream.db'

    conn, c = create_db(db_name)

    collected_users_list = []
    tweets_count = 0
    tweets_per_user = 1
    friends_per_user = 2
    limit_depth = 4  # The depth of the tree of users to collect tweets from

    # The screen_nameof the parent user from which children tweeps will be recursed over
    seed_name = "_vampirre"

    # Collects the user object of the seed screen_name
    seed_user = collect_user(twitter_api, seed_name)

    start_time = time.time()
    streamer = TwitterStreamer(twitter_api)
    _ = streamer.stream_from_users(seed_user,
                          tweets_per_user, friends_per_user, limit_depth, conn, c)

    conn.close()

    print('Collected tweets: %d' % tweets_collected)
