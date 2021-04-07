from functools import partial
from http.client import BadStatusLine
from urllib.error import URLError

from db_manipulation import *
from config import collected_users_list, tweets_count, start_time, limit_depth

import sys
import time
import twitter

import numpy as np
import pymongo


def oauth_login():
    """Returns a twitter api object using OAuth login."""
    CONSUMER_KEY = 'agaAIqBTfKThsSHYArhX8Rc42'
    CONSUMER_SECRET = 'vZMmSIWaXhuGZFRbLHg6vb3Gm91SIAQduOwLZudbDaD2Y6rhLB'
    OAUTH_TOKEN = '1248519309887930372-9zztBJDAfiDLjXOXgpaELnDly0Gbkz'
    OAUTH_TOKEN_SECRET = 'Qu35S7qaHLvHgE0EQbxe6RTykcGCm8xj7t4WRvVqIGzwF'

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):
    """ 
    A nested helper function that handles common HTTPErrors. 

    Returns:
    1. an updated value for wait_period if the problem is a 500 level error. 
    2. Block until the rate limit is reset if it's a rate limiting issue (429 error). 
    3. None for 401 and 404 errors, which requires special handling by the caller.
    """
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):

        if wait_period > 3600:  # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e

        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes

        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None

        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None

        elif e.e.code == 429:
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)

            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60*15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2

            else:
                raise e  # Caller must handle the rate limiting issue

        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds'
                  .format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period

        else:
            raise e

    # End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            return twitter_api_func(*args, **kw)

        except twitter.api.TwitterHTTPError as e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)

            if wait_period is None:
                return

        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("URLError encountered. Continuing.", file=sys.stderr)

            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise

        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("BadStatusLine encountered. Continuing.", file=sys.stderr)

            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)


def stream_from_users(twitter_api, user, tweets_per_user, friends_per_user, mongo_db, depth):
    """ 
    Stores user's tweets then recurses over their friends.    

    Args:
    1. twitter_api (object): OAuth login object for twitter api.
    2. user (object): User object to collect tweets from.
    3. tweets_per_user (int): The number of tweets to get from each user.
    4. friends_per_user (int): The number of friends to recurse over.
    5. tweets_left (int): The total number of tweets to be collected from this user and recursive calls over
                        friends.
    6. db_cursor (file): cursor of database to store the tweets and users collected.

    Returns:
    The function doesn't return any value.

    """

    # avoid duplicates by storing collected users in a global list
    global collected_users_list
    global tweets_count  # global variable to increment count of tweets collected
    global limit_depth  # global variable to limit the recursion
    global start_time  # global variable to calculate the duration

    try:
        tweets = collect_tweets(twitter_api, user, tweets_per_user)
    except ValueError:
        print("Encountered JSONDecodeError with user: {}. Trying again.", user['screen_name'], file=sys.stderr)
        try:
            tweets = collect_tweets(twitter_api, user, tweets_per_user)
        except ValueError:
            print("Encountered JSONDecodeError again", file=sys.stderr)
            
    if tweets:
        _ = save_to_mongo(tweets, mongo_db, "tweets")
        _ = save_to_mongo(user, mongo_db, "users")


    tweets_count += len(tweets)
    collected_users_list.append(user['id'])

    if len(collected_users_list) % 10 == 0:
        print("Collected users: %d user\nCollected tweets: %d tweet\nDuration: %d seconds" %
              (len(collected_users_list), tweets_count, time.time() - start_time), "\n", file=sys.stderr)

    if depth < limit_depth:
        try:
            friends = collect_friends(twitter_api, user, friends_per_user)
            followers = collect_followers(twitter_api, user, friends_per_user)
        except ValueError:
            print("Encountered JSONDecodeError. Trying again...", file=sys.stderr)
            try:
                friends = collect_friends(twitter_api, user, friends_per_user)
                followers = collect_followers(twitter_api, user, friends_per_user)
            except ValueError:
                print("Encountered JSONDecodeError", file=sys.stderr)
                friends, followers = [], []

        # in case no friends or followers were returned by the collecting functions
        if friends and followers:

            # randomly select n friends and followers from the merged lists
            friends = np.random.choice(friends+followers, friends_per_user)

            for friend in friends:
                if friend['id'] not in collected_users_list:
                    _ = stream_from_users(twitter_api, friend, tweets_per_user, friends_per_user,
                                          mongo_db, depth+1)
    # else:
    #     print("Collected tweets: %d tweet\nDuration: %d seconds" %
    #           (tweets_count, time.time() - start_time), "\n")


def collect_tweets(twitter_api, user, tweets_per_user):
    """Returns a number of tweet objects from a user."""
    robust_collect_tweets = partial(
        make_twitter_request, twitter_api.statuses.user_timeline)

    kw = {
        'count': 200,
        'trim_user': 'true',
        'include_rts': 'false',
        'since_id': 1 ,
        'user_id': user['id']
    }


    max_pages = 16
    results = []

    tweets  = robust_collect_tweets(**kw)

    if tweets is None:
        tweets = []

    results += tweets
    print('Fetched {0} tweets for {1}'.format(len(tweets), user['id']), file=sys.stderr)
    page_num = 1


    # tested increasing tweets_per_user to 300 for example, and the result was 3 requests that didn't achieve
    # the required number of 200. So better to set it this way for now to prevent bleeding requests for extra
    # 50 tweets.
    if tweets_per_user == kw['count']:
        page_num = max_pages

    while page_num < max_pages and len(tweets) > 0 and len(results) < tweets_per_user:
        kw['max_id'] = min([tweet['id'] for tweet in tweets]) - 1
        tweets = robust_collect_tweets(**kw)
        results += tweets
        print('Fetched {0} tweets for {1}'.format(len(tweets), user['id']), file=sys.stderr)

    print('Done fetching tweets for {0}\n'.format(user['id']), file=sys.stderr)
    
    return results[:tweets_per_user]


def collect_friends(twitter_api, user, friends_per_user):
    """Return a list of user's friends."""
    robust_collect_friends = partial(
        make_twitter_request, twitter_api.friends.list)

    friends = []
    cursor = -1
    while cursor != 0:
        response = robust_collect_friends(user_id=user['id'], count=200)
        if response is not None:
            friends += response['users']
            cursor = response['next_cursor']

        if len(friends) >= friends_per_user or response is None:
            break
    if friends:
        friends = [user_obj for user_obj in friends if not user_obj['verified']]

    return friends

def collect_followers(twitter_api, user, friends_per_user):
    """Return a list of user's follwoers."""
    robust_collect_followers = partial(
        make_twitter_request, twitter_api.followers.list)

    followers = []
    cursor = -1
    while cursor != 0:
        response = robust_collect_followers(user_id=user['id'], count=200)
        if response is not None:
            followers += response['users']
            cursor = response['next_cursor']

        if len(followers) >= friends_per_user or response is None:
            break

    if followers:
        followers = [user_obj for user_obj in followers if not user_obj['verified']]

    return followers


def collect_user(twitter_api, screen_name):
    """Returns a user object."""
    robust_collect_user = partial(make_twitter_request, twitter_api.users.show)
    return robust_collect_user(screen_name=screen_name)

def collect_users(twitter_api, screen_names):
    robust_collect_users = partial(make_twitter_request, twitter_api.users.lookup)
    return robust_collect_users(screen_name=screen_names)


def store_tweets(twitter_api, tweets, user, db, db_cursor):
    """Stores tweets into a sql database."""
    for tweet in tweets:
        if 'retweeted_status' in tweet.keys():  # check if the tweet was retweeted
            tweet_text = tweet['retweeted_status']['full_text']
            tweet_user = tweet['retweeted_status']['user']['screen_name']

            # collect and store the original author of the tweet
            original_user = collect_user(twitter_api, tweet_user)
            _ = store_user(original_user, db, db_cursor)
        else:
            tweet_text = tweet['full_text']
            tweet_user = tweet['user']['screen_name']
        db_cursor.execute("INSERT INTO tweets VALUES(?, ?, ?)",
                          (tweet_text, tweet['id'], tweet_user))
        db.commit()

def save_to_mongo(data, mongo_db, mongo_db_coll):
    client = pymongo.MongoClient('localhost', 27017)
    db = client[mongo_db]
    coll = db[mongo_db_coll]

    return coll.insert(data)   


def store_tweets_json(tweets, user):
    print(user['id'])
    filename = 'user_tweets/{}.json'.format(user['id'])
    with open(filename, 'w') as f:
        json.dumps(tweets, f, ensure_ascii=False)


def store_user(user, db, db_cursor):
    """Stores user into a sql database."""
    db_cursor.execute("INSERT INTO users VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                      (user['screen_name'], user['name'], user['id'], user['description'],
                          user['followers_count'], user['friends_count'], user['statuses_count'],
                          user['created_at']))
    db.commit()


# Define a twitter streamer object in order to control it's behavior as an instance with defined characterstic
class TwitterStreamer:
    """
    A class representation of tweets streamer with self-contained attributes to avoid setting global variables 
    between different modules.
    """
    def __init__(self, twitter_api):
        self.twitter_api = twitter_api

    def stream_from_users(self, seed_user, tweets_per_user, friends_per_user, limit_depth, db, db_cursor):
        self.collected_users_list = []
        self.tweets_count = 0
        self.tweets_per_user = 1
        self.friends_per_user = 2
        self.limit_depth = limit_depth  # The depth of the tree of users to collect tweets from
        self.db = db
        self.db_cursor = db_cursor
        self._stream_from_user_helper(seed_user, 0)

    def _stream_from_user_helper(self, user, depth):
        tweets = collect_tweets(self.twitter_api, user, self.tweets_per_user)

        if tweets:
            _ = store_tweets(self.twitter_api, tweets, user, self.db, self.db_cursor)
            _ = store_user(user, self.db, self.db_cursor)

        self.tweets_count += len(tweets)
        self.collected_users_list.append(user['id'])

        if len(self.collected_users_list) % 100 == 0:
            print("Collected tweets: %d tweet\nDuration: %d seconds" %
                  (self.tweets_count, time.time() - start_time), "\n")

        if depth < self.limit_depth:
            friends = collect_friends(self.twitter_api, user, self.friends_per_user)

            for friend in friends:

                if friend['id'] not in self.collected_users_list:
                    self.stream_from_user_helper(friend, depth+1)