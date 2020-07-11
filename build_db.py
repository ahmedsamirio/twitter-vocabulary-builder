import os
import sqlite3
from twitter_api import *


if __name__ == "__main__":

    twitter_api = oauth_login()

    conn, c = create_db(db_name)

    # Collects the user object of the seed screen_name
    seed_user = collect_user(twitter_api, seed_name)

    # streamer = TwitterStreamer(twitter_api)
    # _ = streamer.stream_from_users(seed_user
    #                       tweets_per_user, friends_per_user, limit_depth, conn, c)

    stream_from_users(twitter_api, seed_user, tweets_per_user, friends_per_user, conn, c, 0)

    conn.close()

    print('Collected tweets: %d' % tweets_count)
