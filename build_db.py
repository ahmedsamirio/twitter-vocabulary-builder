import time
import sys
from twitter_api import *
from config import *

if __name__ == "__main__":

    twitter_api = oauth_login()

    print('Collecting users and tweets...\n', file=sys.stderr)
    # Collects the user object of the seed screen_name
    seed_users = collect_users(twitter_api, seed_names)

    for user in seed_users:
        stream_from_users(twitter_api, user, tweets_per_user, friends_per_user, mongo_db, 0)

    print('Scraping finished.', file=sys.stderr)