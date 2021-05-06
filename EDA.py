# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Import libraries

# +
import pymongo
from pprint import pprint
from twitter_api import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pytz
import datetime
from dateutil.relativedelta import relativedelta

seed = 44
np.random.seed(seed)

first_pass = False
# -

# # Connect to MongoDB client

client = pymongo.MongoClient('localhost', 27017)
client.list_database_names()

# # Select database and view collections

db = client['twitter_stream_2']
db.list_collection_names()

db.command("dbstats")

# # Explore the structure of documents in each collection
# ## Query one tweet

db.tweets.find_one()

# querying tweets for certain user
count = 0
limit = 10
for tweet in db.tweets.find({'user.id': 275644099}):
    if count > limit:
        break
    pprint(tweet)
    count += 1

# ## Query one user

db.users.find_one()

# In order to aggregate features based on tweets types, we need to first aggregate based over all original tweets (and maybe replies), and then aggregate the percentage of original tweets, replies, and retweets from the total number of tweets collected for every user.
#
# We can also aggregate the frequency of original tweets and retweets by calculating the time between the first and last tweet collected.

# ## Count of users in the users and tweets collection

print('Total number of users in users collection:', len(db.users.distinct('id')))
print('Total number of users in tweets collection:', len(db.tweets.distinct('user.id')))

# # Converting users collection into Pandas dataframe

# +
import pandas as pd

if first_pass:
    users_df = pd.DataFrame(list(db.users.find()))
else:
    users_df = pd.read_csv('users.csv', lineterminator='\n')

users_df.head()
# -

# A user object contains a lot of valuable information about a user, most importantly it shows their:
#     1. followers count
#     2. friends count
#     3. statuses count
#     4. favourites count
#     5. creation date
#     
# Of course there are alot of more features that could be analyzed like names, screen names, descriptions, location, etc..
#
# But for this analysis I'll focus on these 5 features and engineer new features based on the tweets collected for each of these users.

users_df.sample(10)

# ## Extracting the number of months since account creation

# +
# if first_pass:
#     utc=pytz.UTC

#     now = utc.localize(datetime.datetime.now())

#     users_df['delta'] = pd.to_datetime(users_df.created_at).apply(lambda x: relativedelta(now, x))
#     users_df['years'] = pd.to_datetime(users_df.created_at).apply(lambda x: relativedelta(now, x).years)
#     users_df['months'] = users_df.delta.apply(lambda x: x.years * 12 + x.months)
#     users_df['days'] = users_df.delta.apply(lambda x: x.years * 365 + x.days)
# -

# ## Engineering new aggregative features based on tweets collection

# ### Aggregating features for all tweets per user (including retweets and replies)

# Testing aggregating over the latest 100 tweets only and not 200 tweets since after exploring the data, it turns out that I was only able to collect 200 tweets for half of the users.
#
# Update: This is the prototype query for which all of the other queries shall be replicated from

# +
agg_all = db.tweets.aggregate([
#     {
#         '$match': {
#             'user.id': 422
#         }
#     },
    {
        '$group':{
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id',
                'in_reply_to_status_id': '$in_reply_to_status_id',
                'created_at': {
                    '$dateFromString': {
                        'dateString': '$created_at'
                    },
                },
            },
            'count': {'$sum': 1},
        }
    },
    {
        '$sort': {
            '_id.created_at': 1
        }
    },
    {
        '$group': {
            '_id': '$_id.user_id',
            'tweets': {
                '$push': {
                    'tweet_id': '$_id.tweet_id',
                    'created_at': '$_id.created_at',
                    'in_reply_to_status_id': '$_id.in_reply_to_status_id'
                }
            }
        }
    },
    {
        '$project': {
            'tweets': {
                '$slice': ['$tweets', 100]
            }
        }
    },
    {
        '$match': {
            'tweets.in_reply_to_status_id': {
                '$ne': None
            }
        }
    },
    {
        '$project': {
            'count':{'$size': '$tweets'},
            'min_date': {'$min': '$tweets.created_at'},
            'max_date': {'$max': '$tweets.created_at'},
        }
    }
#     {
#         '$group':{
#             '_id': {
#                 'user_id':'$_id.user_id',
#             },
#             'count': {'$sum': 1},
#             'min_tweet_time': {'$min': '$_id.created_at'},
#             'max_tweet_time': {'$max': '$_id.created_at'}
#         }
#     },
#     {
#         '$sort': {
#             '_id.user_id': 1
#         }
#     }
    ], allowDiskUse=True
)

for i, r in enumerate(agg_all):
    print(r)
    if i > 10: break

# +
agg_all = db.tweets.aggregate([
    {
        '$match': {
            'user.id': 18735040,
            'in_reply_to_status_id': {
                '$ne': None
            }
        }
    },
    {
        '$group':{
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id',
                'created_at': {
                    '$dateFromString': {
                        'dateString': '$created_at'
                    },
                },
            },
            'count': {'$sum': 1},
        }
    },
    {
        '$sort': {
            '_id.created_at': 1
        }
    },
    {
        '$group':{
            '_id': {
                'user_id':'$_id.user_id',
            },
            'count': {'$sum': 1},
            'min_tweet_time': {'$min': '$_id.created_at'},
            'max_tweet_time': {'$max': '$_id.created_at'}
        }
    },
#     {
#         '$sort': {
#             '_id.user_id': 1
#         }
#     }
    ], allowDiskUse=True
)

for i, r in enumerate(agg_all):
    print(r)
    if i > 10: break
# -

agg_all = db.tweets.aggregate([
    {
        '$group':{
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id',
                'created_at': {
                    '$dateFromString': {
                        'dateString': '$created_at'
                    },
                },
            },
            'count': {'$sum': 1},
        }
    },
    {
        '$group':{
            '_id': {
                'user_id':'$_id.user_id',
            },
            'count': {'$sum': 1},
            'min_tweet_time': {'$min': '$_id.created_at'},
            'max_tweet_time': {'$max': '$_id.created_at'}
        }
    },
    {
        '$sort': {
            '_id.user_id': 1
        }
    }
    ], allowDiskUse=True
)

# ### Aggregating features for replies

agg_rp = db.tweets.aggregate([
    {
        '$match':{
            'retweeted_status': None,
            'in_reply_to_status_id': {
                '$ne': None
            }
        }
    },
    {
        '$group':{
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id',
                'created_at': {
                    '$dateFromString': {
                        'dateString': '$created_at'
                    },
                },
                'in_reply_to_status_id': '$in_reply_to_status_id'
            },
            'count': {'$sum': 1},
            'retweet_count': {'$max': '$retweet_count'},
            'favorite_count': {'$max': '$favorite_count'},
        }
    },
    {
        '$group':{
            '_id': {
                'user_id':'$_id.user_id',
            },
            'count': {'$sum': 1},
            'avg_rt': {'$avg': '$retweet_count'},
            'avg_fv': {'$avg': '$favorite_count'},
            'total_rt': {'$sum': '$retweet_count'},
            'total_fv': {'$sum': '$favorite_count'},
            'min_tweet_time': {'$min': '$_id.created_at'},
            'max_tweet_time': {'$max': '$_id.created_at'}
        }
    },
    {
        '$sort': {
            '_id.user_id': 1
        }
    }
    ], allowDiskUse=True
)

# ### Aggregating features for original tweets per user (excluding replies)

agg_org = db.tweets.aggregate([
    {
        '$match':{
            'retweeted_status': None,
            'in_reply_to_status_id': None
        }
    },
    {
        '$group':{
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id',
                'created_at': {
                    '$dateFromString': {
                        'dateString': '$created_at'
                    },
                },
            },
            'count': {'$sum': 1},
            'retweet_count': {'$max': '$retweet_count'},
            'favorite_count': {'$max': '$favorite_count'},
        }
    },
    {
        '$group':{
            '_id': {
                'user_id':'$_id.user_id',
            },
            'count': {'$sum': 1},
            'avg_rt': {'$avg': '$retweet_count'},
            'avg_fv': {'$avg': '$favorite_count'},
            'total_rt': {'$sum': '$retweet_count'},
            'total_fv': {'$sum': '$favorite_count'},
            'min_tweet_time': {'$min': '$_id.created_at'},
            'max_tweet_time': {'$max': '$_id.created_at'}
        }
    },
    {
        '$sort': {
            '_id.user_id': 1
        }
    }
    ], allowDiskUse=True
)

# ### Aggregating features for retweets per user

agg_rt = db.tweets.aggregate([
    {
        '$match':{
            'retweeted_status': {
                '$ne': None
            }
        }
    },
    {
        '$group':{
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id',
                'created_at': {
                    '$dateFromString': {
                        'dateString': '$created_at'
                    },
                },
            },
            'count': {'$sum': 1},
            'retweet_count': {'$max': '$retweet_count'},
            'favorite_count': {'$max': '$retweeted_status.favorite_count'},
        }
    },
    {
        '$group':{
            '_id': {
                'user_id':'$_id.user_id',
            },
            'count': {'$sum': 1},
            'avg_rt': {'$avg': '$retweet_count'},
            'avg_fv': {'$avg': '$favorite_count'},
            'total_rt': {'$sum': '$retweet_count'},
            'total_fv': {'$sum': '$favorite_count'},
            'min_tweet_time': {'$min': '$_id.created_at'},
            'max_tweet_time': {'$max': '$_id.created_at'}
        }
    },
    {
        '$sort': {
            '_id.user_id': 1
        }
    }
    ], allowDiskUse=True
)

if first_pass:
    cursor_name = [(agg_all, 'all', 'Added aggregated features of all tweets for'),
                   (agg_rp, 'rp', 'Added aggregated features of replies for'),
                   (agg_org, 'org', 'Added aggregated features of original tweets for'),
                   (agg_rt, 'rt', 'Added aggregated features of retweeted tweets for')]

    print('Adding aggregated features...')
    for agg, ft, msg in cursor_name:
        count = 0
        for i in agg:
            count += 1
            user_id = i['_id']['user_id']
            index = users_df.query('id == @user_id').index
            users_df.loc[index, f'tweets_count_{ft}'] = i['count']
            if ft != 'all':
                users_df.loc[index, f'avg_favorites_{ft}'] = i['avg_fv']
                users_df.loc[index, f'avg_retweets_{ft}'] = i['avg_rt']
                users_df.loc[index, f'total_favorites_{ft}'] = i['total_fv']
                users_df.loc[index, f'total_retweets_{ft}'] = i['total_rt']
            users_df.loc[index, f'min_date_{ft}'] = i['min_tweet_time']
            users_df.loc[index, f'max_date_{ft}'] = i['max_tweet_time']
        print(msg, count, 'users.')

if first_pass:
    users_df.to_csv('users.csv', index=False)

# ## Sanity check
# 1. Check that the total number of users that have replies in the users collection is 3325
# 2. Check that the total number of users that have original tweets in the users collection is 3673
# 3. Check that the total number of users that have retweeted tweets in the users collection is 3361

# +
# 1 
distinct_rp = db.tweets.aggregate([
    {
        '$match': {
            'in_reply_to_status_id': {
                '$ne': None
            }
        }
    },
    {
        '$group': {
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id'
            },
            'count': {'$sum': 1}
        }
    },
    {
        '$group': {
            '_id': {
                'user_id': '$_id.user_id',
            },
            'count': {'$sum': 1}
        }
    },
    {
        '$group': {
            '_id': None,
            'count': {'$sum': 1}
        }
    }
])

assert list(distinct_rp)[0]['count'] == 3325

# +
# 2
distinct_org = db.tweets.aggregate([
    {
        '$match': {
            'in_reply_to_status_id': None,
            'retweeted_status': None
        }
    },
    {
        '$group': {
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id'
            },
            'count': {'$sum': 1}
        }
    },
    {
        '$group': {
            '_id': {
                'user_id': '$_id.user_id',
            },
            'count': {'$sum': 1}
        }
    },
    {
        '$group': {
            '_id': None,
            'count': {'$sum': 1}
        }
    }
])

assert list(distinct_org)[0]['count'] == 3673

# +
# 2
distinct_rt = db.tweets.aggregate([
    {
        '$match': {
            'retweeted_status': {
                '$ne': None
            }
        }
    },
    {
        '$group': {
            '_id': {
                'user_id': '$user.id',
                'tweet_id': '$id'
            },
            'count': {'$sum': 1}
        }
    },
    {
        '$group': {
            '_id': {
                'user_id': '$_id.user_id',
            },
            'count': {'$sum': 1}
        }
    },
    {
        '$group': {
            '_id': None,
            'count': {'$sum': 1}
        }
    }
])

assert list(distinct_rt)[0]['count'] == 3361
# -

users_df.columns[-24:]

old_feats = ['friends_count', 'followers_count', 'statuses_count', 'favourites_count']
new_feats = ['tweets_count_all', 'tweets_count_rp', 'avg_favorites_rp', 'avg_retweets_rp',
             'total_favorites_rp', 'total_retweets_rp', 'tweets_count_org', 'avg_favorites_org', 
             'avg_retweets_org', 'total_favorites_org', 'total_retweets_org', 'tweets_count_rt',
             'avg_favorites_rt', 'avg_retweets_rt', 'total_favorites_rt', 'total_retweets_rt']

users_df[old_feats + new_feats]= users_df[old_feats + new_feats].fillna(0)

df = users_df[old_feats + new_feats]

df.hist(figsize=(15, 15), bins=50);

# ## What are the question that I want to answer?
# 1. How long does it take to tweet 200 tweets?
# 2. How do users in different follower groups use twitter?
# 3. What are the characterstics of users who don't retweet and those who do?
# 4. Do users who interact more commonly throught replies have more or less followers?
# 5. Is the frequency of original tweets or retweeted tweets correlated with the number of followers?
# 6. What is the proportion of passive users? (users who didn't post original tweets in the latest 200 tweets)
# 7. Do veteran users retweet often to less?
# 8. How many users were collected with their accounts created only in the past week?
# 9. Is more retweets correlated with more followers?
# 10. Does the total number of retweets of tweets the user has retweeted correlate with his followers count?
# 11. Can we deduce if users have reduced using the platform, or started using it more often?
# 12. How does the two groups differ? Does some get bored or discouraged because they don't get many followers? And does the other group get encouraged because of a spike of followers that they have?
# 13. Is there a relationship between description length and followers count?
# 14. Label users based on churning and ask more questions

# ## 1. How long does it take to tweet 200 tweets?
#
# The answer to this question lies in calculating the relative difference between the date of the earliest and the date of collecting the date which is 19/4/2021

# convert dates to datetime objects
users_df['min_date_all'] = pd.to_datetime(users_df['min_date_all'])
users_df['max_date_all'] = pd.to_datetime(users_df['max_date_all'])
users_df['min_date_org'] = pd.to_datetime(users_df['min_date_org'])
users_df['max_date_org'] = pd.to_datetime(users_df['max_date_org'])
users_df['min_date_rt'] = pd.to_datetime(users_df['min_date_rt'])
users_df['max_date_rt'] = pd.to_datetime(users_df['max_date_rt'])
users_df['min_date_rp'] = pd.to_datetime(users_df['min_date_rp'])
users_df['max_date_rp'] = pd.to_datetime(users_df['max_date_rp'])
users_df['created_at'] = pd.to_datetime(users_df['created_at']).apply(lambda x: x.replace(tzinfo=None))
collection_date = users_df['max_date_all'].max()

# calculate difference in times between min and max dates
tweets_duration = users_df.apply(lambda x: (collection_date - x.min_date_all).days, axis=1)

# Now since the duration of tweeting 200 tweets for each user was collected, we weren't able to collect 200 tweets for every user, let's take a look at the distribution of tweets collected.

sns.histplot(data=df, x='tweets_count_all');
plt.title('Distribution of number of tweets collected per user');

# Let's also take a look at 5 number summary of the feature to determine how many users fit our criteria.

df['tweets_count_all'].describe()

# We have it that we collected more than 148 tweets for 75% of the users, and more than 198 tweets for 50% of the users. Let's take a look.

q_50 = df['tweets_count_all'] > df['tweets_count_all'].quantile(0.5)
q_25 = df['tweets_count_all'] > df['tweets_count_all'].quantile(0.25)

# UPDATE: I thought about this step again, and destroyed a part of the data that actually didn't tweet at all and was maybe created a while ago. Think about, if we only look at users for whom we have collected 200 tweets, the users for whom we have collected less than that number, and they didn't tweet more than it in their lifetime won't be represented. Let's take a look at that.

users_df.query('tweets_count_all < 199 and statuses_count == tweets_count_all').shape[0]

# So as we see, there are 699 users whom we have dropped for no reason, also let's take a look at the number of users remaining.

mask = (users_df[~q_50].tweets_count_all < 199) & (users_df[~q_50].statuses_count == users_df[~q_50].tweets_count_all)
rem_users = users_df[~q_50][~mask]
print('Number of remaining users', rem_users.shape[0])
print('Proportion of remaining users {:.2f}'.format(rem_users.shape[0]/users_df.shape[0]))

# So even if do it this way, we'd be dropping 37% of the data. I don't think that this is particularly a right thing to do. Since we already have collected tweets for said users, they shall be used in context of frequencies, as we have certain statistics based on their recent N tweets. 
#
# Let's take a look at the distribution of their tweets count.

rem_users.tweets_count_all.plot(kind='box');

# We have it that the majority of their tweets count is above 120, and the rest are outliers. But how many?

drop = (rem_users.tweets_count_all < 120).index


# 251 users is definetly better than dropping 1450 users, and more over dropping them could be the mask that we want instead of dropping more than 50% of the data.
#
# Since we opted to keep that data that we have, asking the question about the duration that it takes to tweets 200 tweets isn't valid, because we don't have 200 tweets for all users. But we have is that date of the earliest tweet, and the date of the latest tweet, and the date of collection, so we can calculate a frequency of tweets per day instead of 200 tweets, then we can calculate the duration it takes any user to tweet N tweets for example. 
#
# Let's take a look.

def calculate_frequency(row, count_feat, duration_feat):
    """Calculates frequency for a total value over a duration per pandas df row."""
    if row[duration_feat] == 0: 
        return row[count_feat] 
    else:
        return row[count_feat] / (row[duration_feat] + 1)


users_df['tweets_min_coll_duration'] = users_df.apply(lambda x: (collection_date - x.min_date_all).days, axis=1)
users_df['tweets_min_max_duration'] = users_df.apply(lambda x: (x.max_date_all - x.min_date_all).days, axis=1)
users_df['days_since_creation'] = users_df.apply(lambda x: (collection_date - x.created_at).days, axis=1)

# We calculated the duration from two perspectives, the first is the difference in days between the earliest collected tweet and the day of tweets collection, and the second is the differene in days between the earliest and latest tweet collected. I think that the first will be more accurate since it will account for some edge cases where a user may have stopped tweeting for a long time, and so there will be a discrepancy betweent the date of the latest tweet and the date of collection.
#
# We can also calculate another frequency based on the statuses count per user and the difference in days between their creation date and the date of collection.

# +
plt.figure(figsize=(15, 5))
plt.suptitle('Duration between')

plt.subplot(1, 3, 1)
sns.histplot(users_df.tweets_min_coll_duration);
plt.title('earliest tweet and collection date');
plt.xlabel('Duration in days')

plt.subplot(1, 3, 2)
sns.histplot(users_df.tweets_min_max_duration);
plt.title('earliest tweet and latest tweet');
plt.xlabel('Duration in days')


plt.subplot(1, 3, 3)
sns.histplot(users_df.days_since_creation);
plt.title('account creation and collection date');
plt.xlabel('Duration in days');

# -

# The difference in duration between the first two plot could be used to find users who stopped using twitter.

plt.figure(figsize=(10, 5))
plt.scatter(x=users_df.tweets_min_coll_duration,
            y=users_df.tweets_min_max_duration,
            s=users_df.statuses_count/1000,
            alpha=0.3);
plt.xlabel('Duration between earliest tweet and collection')
plt.ylabel('Duration between earliest tweet and latest')
plt.plot([0, 4000], [0, 4000], linestyle='--', alpha=0.4, color='r');

# The regression like line is the users who didn't quit twitter. The more on the left a user is on this line, the more active he/she is, and this is indicated also by their statuses count, and the more on the right, the less the activity of this users, despite not quitting twitter.
#
# So for example we have users who have a difference of 3000 days between their earliest tweet and date of collection, and the same for earliest and latest tweets, so probably those would be users who came back from the dead.
#
# Then we have users who fall out of line, and these are users who have duration between earliest tweet and collection which is larger than duration between earliest and latest tweet.
#
# So let's get back to the main question
#
# ## How many tweets per day does a typical twitter user tweet?
#
# Since we have 3 different durations, we could calculate 3 different frequencies. The first one would give the most accurate in terms of recent times, the second on would give a good representation frequency of users who quit before quitting, and the third one would give the over time average each user has, let's take a look at all of them. 

users_df['tweet_freq_min_coll'] = users_df.apply(lambda x: calculate_frequency(x,'tweets_count_all','tweets_min_coll_duration'), axis=1)
users_df['tweet_freq_min_max'] = users_df.apply(lambda x: calculate_frequency(x,'tweets_count_all','tweets_min_max_duration'), axis=1)
users_df['tweet_freq_cr_coll'] = users_df.apply(lambda x: calculate_frequency(x,'statuses_count','days_since_creation'), axis=1)

# +
plt.figure(figsize=(15, 5))
plt.suptitle('Tweet frequency between')

plt.subplot(1, 3, 1)
sns.histplot(users_df.tweet_freq_min_coll);
plt.title('earliest tweet and collection date');
plt.xlabel('Tweets per day')

plt.subplot(1, 3, 2)
sns.histplot(users_df.tweet_freq_min_max);
plt.title('earliest tweet and latest tweet');
plt.xlabel('Tweets per day')

plt.subplot(1, 3, 3)
sns.histplot(users_df.tweet_freq_cr_coll);
plt.title('account creation and collection date');
plt.xlabel('Tweets per day');

# -

# ### Calculate mean based on the three features, and compare them to tease out users with increased activity, decreased activity and users who quit.

print("Median tweets per day calculated using earliest tweet and collection date: {:.2f}".format(users_df.tweet_freq_min_coll.median()))
print("Median tweets per day calculated using earliest tweet and latest tweet date: {:.2f}".format(users_df.tweet_freq_min_max.median()))
print("Median tweets per day calculated using statuses count and collection date: {:.2f}".format(users_df.tweet_freq_cr_coll.median()))

# The increase in 3rd and 2nd median over the 1st indicate the hypothesis that alot of users had high activity, then this activity decreased and then they quit.

plt.figure(figsize=(10, 5))
plt.scatter(x=users_df.tweet_freq_min_coll,
            y=users_df.tweet_freq_min_max,
#             s=users_df.followers_count/100000,
            alpha=0.3);
plt.xlabel('Tweets frequency (min date - collection date)')
plt.ylabel('Tweets frequency (min date - max date)');
plt.axvline(0, color='r', linestyle='--', alpha=0.3);

# Again, the regression like line indicates the users whose didn't stop tweeting while every other users didn't tweet anything since the last tweet. Also the users present on the red dashed line are the users who quit.

plt.figure(figsize=(10, 5))
plt.scatter(x=users_df.tweet_freq_min_coll,
            y=users_df.tweet_freq_cr_coll,
#             s=users_df.followers_count/10000,
            alpha=0.1);
plt.xlabel('Tweets frequency (min date - collection date)')
plt.ylabel('Tweets frequency (creation date - collection date)');
plt.plot([0, 200], [0, 200], c='r', linestyle='--');

# Any user that falls on that dashed line didn't change his behavior in the last 200 tweets compared to his average behavior since creating his account, while users who below the line increased their activity and users who are above have decreased their activity.
#
# So to answer the question of how long does it take to tweet 200 tweets we could have two answers. The first one is based on the tweeting frequency of the latest tweets, and the other is based on the average tweeting frequency since account creation, and they will be totally different, since the second one is around 2.52x the first.
#
# Before I answer that question I want to look at how much the behavior changed per user. So let's calculate a ratio between the recent behavior and the averaged behavior over account life time. If the the ratio is above 1, then the user has uped his tweeting game, and vice versa.

# +
users_df['tweet_freq_ratio'] = users_df.tweet_freq_min_coll / users_df.tweet_freq_cr_coll
prop_inc = users_df.query('tweet_freq_ratio > 1').shape[0] / users_df.shape[0]
prop_dec = users_df.query('tweet_freq_ratio <= 1').shape[0] / users_df.shape[0]

print('Proportion of users whose tweet frequency increased: {:.2f}'.format(prop_inc))
print('Proportion of users whose tweet frequency decreased or stayed the same: {:.2f}'.format(prop_dec))


# +
def calculate_diff(feature, stat='median'):
    feat_inc = users_df.query('tweet_freq_ratio > 1')[feature]
    feat_dec = users_df.query('tweet_freq_ratio <= 1')[feature]
    if stat == 'median':
        return feat_inc.median() - feat_dec.median()
    elif stat == 'mean':
        return feat_inc.mean() - feat_dec.mean()

followers_diff = calculate_diff('followers_count')
statuses_diff = calculate_diff('statuses_count')
days_diff = calculate_diff('days_since_creation')

print('Difference between median followers count between the first and second groups:', followers_diff)
print('Difference between median statuses count between the first and second groups:', statuses_diff)
print('Difference between median days between the first and second groups:', days_diff)
# -

# So on average, users who have a surge in tweeting frequency are users of less followers count of 1081.5 than users who have a decrease in their tweeting frequency or stay the same. Morever, the first group has an average of 3567 statuses less than the second group. We can also see than users of the first group have been on average 545.5 less than users of the second group. 
#
# To understand this, we could say that the second group includes the users who have been on twitter for a while longer than the first group. They have a solid base of followers, and maybe they don't have to prove themselves or don't want anything more from twitter, while the first group's users are increasing their usage of the network in an attempt to increase their followers count.
#
# So maybe the first group are tweeting as much as they can in order to have a better chance at getting retweets and so increase their followers count. 

# So again, to answer the main question, how long does it take, on average, a user on twitter to tweet 200 tweets?
#
# Based on the collected data it takes around 341 days, which would be extremely biased because as we have seen, a lot of users had decreased tweeting frequency in the collected tweets. While based on their statuses count and the time since creation it would take around 134 days.

# It would be interesting to look the tweeting habits dissected between groups of users with different followers count, so let's bin the followers count into different bins and look into them.

# bins followers count
bins = [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 200000000]
followers_labels = ['0-10', '10-100', '100-1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '>=10M']
users_df['followers_binned'] = pd.cut(users_df.followers_count, bins=bins, labels=followers_labels, include_lowest=True)

# Let's take a look at the count of each group first.

plt.figure(figsize=(10, 5))
ax = users_df['followers_binned'].value_counts()[followers_labels].plot(kind='bar')
plt.xlabel('Followers group')
plt.ylabel('Count')
plt.title('Users binned into different groups according to followers count')
for j, v in enumerate(users_df['followers_binned'].value_counts()[followers_labels].tolist()):
    ax.text(j-0.17, v+10, str(users_df['followers_binned'].value_counts()[followers_labels][j]))

# If we were to look in descending order of users in each group.

plt.figure(figsize=(10, 5))
followers_prop = users_df['followers_binned'].value_counts().sort_values() / users_df.shape[0]
ax = followers_prop.plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Group')
plt.title('Proportions of users binned into different groups according to followers count')
plt.xticks(np.arange(0, 0.3, 0.02))
for j, v in enumerate(followers_prop.tolist()):
    ax.text(v+0.001, j-0.15, str(users_df['followers_binned'].value_counts().sort_values()[j]))

# 25% of the users fall in the 100-1K group, while 20% fall in the following 1K-10K, and around 15% are in the 10-100 group. 14% of the users fall in the 10K-100K, and 9% of users fall in 100K-1M, and 8% are in 0-10, while around 6% ar in 1M-10M, and only 1% are above 10M.

# +
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
users_df.groupby('followers_binned')['tweet_freq_min_coll'].median().plot(kind='bar');
plt.title('Latest tweets frequency since tweets collection')
plt.xlabel('Followers bin')
plt.ylabel('Tweets per day');

plt.subplot(1, 3, 2)
users_df.groupby('followers_binned')['tweet_freq_cr_coll'].median().plot(kind='bar');
plt.title('Average tweets frequency since account creation')
plt.xlabel('Followers bin')
plt.ylabel('Tweets per day');

plt.subplot(1, 3, 3)
users_df.groupby('followers_binned')['tweet_freq_ratio'].median().plot(kind='bar');
plt.title('Frequency ratio')
plt.xlabel('Followers bin')
plt.ylabel('Tweets per day');
# -

# We can see that their is a general decrease in the latest activity over all groups.

# ## 2. How do users in different follower groups use twitter?
#
# We can look into different facets, the first one is to look at the proportion of original tweets, retweets and replies for every group. The second one is to look at how the content they retweet differ between each group. The third one is find how many favorites per day do they have? As an indicator of how much time they spend looking at their timeline.
#

# +
#TODO: plot a barplot with the original tweets on the upper axis and retweets (or retweets + replies) on the
# lower axis for each group

# calculate proportion of original tweets from all tweets collected per user
users_df['tweets_prop_org'] = users_df['tweets_count_org'] / users_df['tweets_count_all']
users_df['tweets_prop_rt'] = users_df['tweets_count_rt'] / users_df['tweets_count_all']
users_df['tweets_prop_rp'] = users_df['tweets_count_rp'] / users_df['tweets_count_all']
# -

plt.figure(figsize=(15, 5))
users_df.groupby('followers_binned')['tweets_prop_org'].median().plot(kind='bar')
(users_df.groupby('followers_binned')['tweets_prop_rt'].median()*-1).plot(kind='bar', color='r');

# Users with followers above 10K tend to retweet less than other groups.

# +
#TODO: plot a barplot with the average number of retweets that tweet a group retweets has and the same for 
#favorites

plt.figure(figsize=(10, 5))
users_df.groupby('followers_binned')['avg_retweets_rt'].median().plot(kind='bar')
# (users_df.groupby('followers_binned')['avg_favorites_rt'].median()*-1).plot(kind='bar', color='r')
# -

# It kind of makes sense that users in >=10M group would have the largest average of retweets and favorites for retweeted tweets, and that's because probably they have a small number of friends, and those friends have a close number of followers, and therefore their tweets have the highest number of tweets and favorites. 
#
# But why the 3 groups from 10K to 10M have low numbers than the rest of the groups, why?
#
# Well, maybe users in these groups mostly retweet tweets from users of lower influence, while users in groups from 10 to 10K followers tend to do the opposite, which is retweet from users in groups from 10K to 10M.

#TODO: calculate favorites per day and calculate the average for each group then barplot it
users_df['favorites_per_day'] = users_df['favourites_count'] / users_df['days_since_creation']

plt.figure(figsize=(15, 5))
users_df.groupby('followers_binned')['favorites_per_day'].median().plot(kind='bar');

# We can see an increase as we go from 0-10 until 1k-10K, which indicates an increase in activity and commitment to use the platform as the followers of a user increase, then as the we move on to the next groups, the favorites per day starts to decrease, which indicate that either the users don't look at the timelines that often, or that they look but they don't have many updates (since they don't follow alot of people), or that they don't view their timeline alltogether.
#
# We can look at the average number of friends per each group to consolidate this.

plt.figure(figsize=(15, 5))
users_df.groupby('followers_binned')['friends_count'].median().plot(kind='bar');

# We can see that as users progress in their followers rank, the number of other users they follow tend to decrease, and therefore the number of activity available to interact with on their timeline is less. So maybe they interact with it, and maybe they don't.
#
# Also, since we saw that the activity of the highest 3 groups tend to be comprised mainly of original tweets, they may be inclined to use twitter as means of tweeting and less of viewing other people's tweets.

# ## How many retweets per original tweet does each group get?

plt.figure(figsize=(15, 5))
users_df.groupby('followers_binned')['avg_retweets_org'].median().plot(kind='bar')

users_df.groupby('followers_binned')['avg_retweets_org'].mean()

# ## 3. What are the characterstics of users who don't retweet and those who do?
#
# Through the last question we concluded that the behavior of a certain user is linked to the number of followers he/she has, and as we move along the groups, their interaction through retweets and favorites tend to decrease. But what if we grouped the users into different groups according to their retweeting habits, what would we find?
#
#
# How can we discern users into different groups according to their retweeting habits? We could split them according to the proportion of retweets collected in their latest tweets as follows, we could say that users who had less than 25% of their latest 200 tweets as retweets as barely retweeting users, while users between 25% and 50% are occasional retweeters, and users betweetn 50% and 75% are moderate retweeters, while the last groups is above 75% which are heavy retweeters.
#
# Or we could split the users by the average proportion of retweets, where users would be assigned to either less than or equal or higher than average.
#
# But that wouldn't take into account the time dimension, as the duration taken between the latest tweets isn't the same for all users. I think that a better strategy is to calculate the number of retweets per day based on the duration of collected tweets for each user.

users_df['retweets_per_day'] = (users_df['tweets_count_rt'] / (users_df['tweets_min_coll_duration']+1))
users_df['retweets_per_day'][users_df['retweets_per_day'] < users_df['retweets_per_day'].quantile(0.95)].hist(bins=5);

# The number of retweets per day significantly condenses users into the first group, and then the rest of the users are distributed over different values, therefore this direction wouldn't be good for analyis, as we can't possibly split the users into representative groups.

users_df['tweets_prop_rt'].hist();

# The users retweets propotion from the latest tweets looks like a better feature to split the users by, so let's do it.

labels = ['Barely', 'Occasionaly', 'Moderatly', 'Heavily']
users_df['tweets_prop_rt_binned'] = pd.cut(users_df['tweets_prop_rt'], labels=labels, bins=4)

users_df['tweets_prop_rt_binned'].value_counts().plot(kind='bar');

# First let's look into followers and friends count.

plt.figure(figsize=(15, 7))
users_df.groupby('tweets_prop_rt_binned')['followers_count'].median().plot(kind='bar');
(users_df.groupby('tweets_prop_rt_binned')['friends_count'].median()*-1).plot(kind='bar', color='r');

# There is no significant differences between groups other than usually users who heavily retweets tend have the lowest number of followers, and a higher number of friends.

# Let's look into average number of retweets for the tweets that they retweet.

plt.figure(figsize=(10, 4))
users_df.groupby('tweets_prop_rt_binned')['avg_retweets_rt'].median().plot(kind='bar');
# (users_df.groupby('tweets_prop_rt_binned')['avg_favorites_rt'].median()*-1).plot(kind='bar', color='r');

# Users who barely retweet have the lowest average of number of retweets for a given retweeted tweet, which is counter intuitive, as we previously concluded that users with huge following don't often retweet, but when they the tweets are hot in terms of retweets and favorites count. 
#
# Let's look at a heatmap that sums up how users in different followers groups are distributed among these 4 retweeting behaviors.

# +
plt.figure(figsize=(12, 6))
retweet_followers = pd.pivot_table(users_df,
                           index='tweets_prop_rt_binned',
                           values='_id', columns='followers_binned', aggfunc=len)

retweet_followers_prop = retweet_followers / retweet_followers.sum(axis=0)
sns.heatmap(retweet_followers_prop,
            cmap="YlGnBu",
            square=True, 
            linewidth=2.5, 
            cbar=False, 
            annot=True,
            fmt=".2f"
           );


# -

# We can see that users are distributed among the 4 categoires, where the majority of the users in all followers groups are users who barely retweets, or users who retweet less than 25% of their total number of tweets.
#
#
# How long have they been on twitter?

(users_df.groupby('tweets_prop_rt_binned')['days_since_creation'].median()/365).plot(kind='bar');

# No pattern.

(users_df.groupby('tweets_prop_rt_binned')['tweets_count_org'].median()).plot(kind='bar');

# What if we split the data into users above and below average of retweeting proportions?

users_df['tweets_prop_rt_above_avg'] = users_df['tweets_prop_rt'] >  users_df['tweets_prop_rt'].median()

users_df['tweets_prop_rt_above_avg'].value_counts().plot(kind='bar');

# Is there a difference in followers count?

users_df.groupby('tweets_prop_rt_above_avg')['followers_count'].median().plot(kind='bar');

# So we have it that users who retweeted less than 18.5% of their latest tweets have an average of 377.5 follower more than users who did.

users_df.groupby('tweets_prop_rt_above_avg')['friends_count'].median().plot(kind='bar');

users_df.groupby('tweets_prop_rt_above_avg')['friends_count'].median().diff()

# On the other hand, users who have above average retweets have an average of 128.5 friends more than users who didn't.

# ## 6. What is the proportion of passive users? (users who didn't post original tweets in the latest 200 tweets)
#

passive_users = users_df.query('tweets_count_org == 0')
print('Proportion of passive users: {:.2f}'.format(passive_users.shape[0]/users_df.shape[0]))

# How are these passive users represented in each followers group?

ax = passive_users.followers_binned.value_counts().plot(kind='bar', figsize=(10, 5));
for j, v in enumerate(passive_users.followers_binned.value_counts().tolist()):
    ax.text(j-0.1, v+0.7, str(passive_users.followers_binned.value_counts()[j]))

# The majority of passive users are users of low followers count, and the users with high are obviously anomalies.

# ## 7. Do veteran users retweet often to less?
#
# How do we define a twitter veteran? Twitter was launched at 2006, and it became popular at 2007. So I'd say that any one joined twitter from 2007 until 2012 and kept being active is a veteran user. So that's more than 9 years ago.

users_df['veteran_flag'] = users_df['years_since_creation'] > 9

users_df.groupby('veteran_flag')['tweets_prop_rt'].median().plot(kind='bar');

# Nope, there is no relationship with being a twitter veteran and how much you retweet.

# ## 8. How many users were collected with their accounts created only in the past week?
#

users_df[users_df['days_since_creation'] <= 6].shape[0]

# ## 9. Is more retweets correlated with more followers?
#
# We actually found out previously the it was the opposite.

# ## 10. Does the total number of retweets of tweets the user has retweeted correlate with his followers count?
#
# I also answered this question previously.

# ## 11. Can we deduce if users have reduced using the platform, or started using it more often?
# Yes and we did.

# ## 12. How does the two groups differ? Does some get bored or discouraged because they don't get many followers? And does the other group get encouraged because of a spike of followers that they have? 
#
# This question require longitudnal data for each user, while the date we have is cross sectional. 
#
# To get an approximation of the effect of this, we can calculate the difference between the duration between earliest and latest tweet collection, and the duration between earliest tweet collected and time of collection.

users_df['duration_diff'] = users_df['tweets_min_coll_duration'] - users_df['tweets_min_max_duration']
users_df['duration_diff'].hist();

# +
inactive = users_df.query('duration_diff >= 365')
plt.scatter(x=inactive['duration_diff'],
            y=inactive['followers_count'])

plt.yscale('log')
plt.ylim([1, 10e7])

# above_30.groupby('followers_binned').duration_diff.mean().plot();
# -

# We can see that as the duration of inactivity increase, the spread of the followers decrease, which indicates that the longer the duration of inactivity, the lower the followers of the inactive users. Therefore we could say that users who quit long time ago probably did because they didn't find their voice heard in twitter, or may be because they didn't have a unique voice.

inactive.followers_binned.value_counts().plot(kind='bar')

# Users above 100K are probably official accounts and these accounts don't necesairly tweet very often. What is interesting is the users between 100 and 100K followers. 
#
# We could tease out how their activity changed in their latest tweets by calculating their tweeting frequency compared to would have been their average tweeting frequency since account creation.

## calculate frequency and compare to min max frequency
inactive['tweets_cr_max_duration'] = inactive.apply(lambda x: ( x.max_date_all - x.created_at).days, axis=1)
inactive['tweets_freq_cr_max'] = inactive['statuses_count'] / (inactive['tweets_cr_max_duration'] + 1)

plt.scatter(x=inactive['tweets_cr_max_duration'],
            y=inactive['tweets_min_max_duration'],
            s=inactive['followers_count']/1000);

inactive['tweet_freq_cr_max'] = inactive.apply(lambda x: calculate_frequency(x,'statuses_count','tweets_cr_max_duration'), axis=1)

plt.scatter(x=inactive['tweet_freq_cr_max'],
            y=inactive['tweet_freq_min_max'],
            s=inactive['followers_count']/100000)
plt.plot([0, 100], [0, 100])
# plt.xlim([0, 100])

inactive['ratio'] = inactive['tweet_freq_min_max'] / inactive['tweet_freq_cr_max']

inactive.groupby('followers_binned').ratio.median().plot(kind='bar');

# The basic permise of the ratio is quite right, as the ratio never exceeds one in the groups we wanted to look at. And it is evident that users in groups between 100 and 100K followers experienced a decrease in activity that compared to their overall behavior.

# ## 13. Is there a relationship between description length and followers count?
#

# First, how many users don't have any description in place?

users_df.description.isna().sum()

desc_len = users_df.description.str.len()
plt.scatter(x=np.log(users_df.followers_count),
            y=desc_len)


plt.hist(desc_len);



# 7. Do veteran users retweet often to less?
# 8. How many users were collected with their accounts created only in the past week?
# 9. Is more retweets correlated with more followers?
# 10. Does the total number of retweets of tweets the user has retweeted correlate with his followers count?
# 11. Can we deduce if users have reduced using the platform, or started using it more often?
# 12. How does the two groups differ? Does some get bored or discouraged because they don't get many followers? And does the other group get encouraged because of a spike of followers that they have?

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
tweets_duration[q_50].hist();
plt.title(f"Duration for tweeting 200 tweets (Above 50th percentile) \nMedian {tweets_duration[q_50].median()} days");
plt.subplot(1, 2, 2)
tweets_duration[q_25].hist();
plt.title(f"Duration for tweeting 200 tweets (Above 25th percentile) \nMedian {tweets_duration[q_25].median()} days");

# # REDACTED
# We can see that the median more than doubles when we include users that we didn't collect 200 tweets for, and that maybe because these users didn't reach 200 tweets in their life time. Let's check that.

df[~q_25].statuses_count.describe()

# # REDACTED
# We can see in the next plot that users below the 25th quantile have statuses counts that generally don't exceed 200 tweets. There are some users have tweeted more than 200 tweets, but we didn't collect the latest ones for all of them, that maybe have happened due to a quirk in the api during collection, but the general rule is that the majority of users for which we didn't collect 200 tweets didn't exceed that number.
#
# Therefore, I think that the best way to estimate the statistic is to use that data above a cutoff point, and the the cutoff of the 50th percentile is illustrated.

df.plot(kind='scatter', x='statuses_count', y='tweets_count_all', figsize=(8, 5))
plt.xscale('log')
plt.axhline(198, color='r', linestyle='--');

# # REDACTED
#
# But why should the cutoff be strictly 198? How will the statistic be affected by choosing a different cutoff. We can calculate the statistic for every single cutoff from 1 to 199 and plot them.

# +
cutoffs = np.arange(1, 200)
statistics = []

for c in cutoffs:
    mask = df.tweets_count_all > c
    statistics.append(tweets_duration[mask].median())
# -

plt.figure(figsize=(14, 5))
plt.plot(cutoffs, statistics);
plt.xlabel('Cutoff')
plt.ylabel('Median duration')
plt.title('Median duration for tweeting 200 tweets over different cut offs');

# # REDACTED
#
# We can see that the slope increases in the last 25 points.

plt.plot(cutoffs, statistics);
plt.xlabel('Cutoff')
plt.ylabel('Median duration')
plt.title('Median duration for tweeting 200 tweets over different cut offs');
plt.xlim([190, 200]);

# # REDACTED
#
# We can that changing the cutoff between 190 and 199 decreases the median duration from 125 to 85 days, which is considerable. 
#
# So I'll stick with the more accurate statistic using 50% of the users. 
#
# On average, a twitter user takes 85 days to tweet 200 tweets, that may include retweets or replies.

# # REDACTED
# ### But how long does it take a user to tweet N original tweets?
#
# The answer to this question isn't as easy as the one before. I'll tell you why but first let's look at the distribution of original tweets collected per user.

sns.histplot(data=df[q_50], x='tweets_count_org');

# The reason I plotted using the 50th quantile mask is that we want to have an accurate picture of the distribution of original tweets collected per user, since we discovered previously that the we weren't able to accurately collect the latest 200 tweets for users who have tweeted more than 200 times in their lifetime.
#
# In order to answer the question with this criteria, we would have to aggregate the earliest date of 100th tweet going back in time for each user, what we can do now is to calculate the average time it takes to tweet a minimum of 100 original tweets. 

tweets_duration_org = users_df.apply(lambda x: (collection_date - x.min_date_org).days, axis=1)

# median duration for tweeting more than 100 and less 200 original tweets
q_50_100 = df[q_50].tweets_count_org >= 100
tweets_duration_org[q_50][q_50_100].median()

# median number of original tweets in collected sample
df[q_50][q_50_100].tweets_count_org.median()

# So on average, it takes a user 165 days to tweet an average of 157 original tweets.

# ### But is this duration related to any other feature? For example the first thing that comes to mind is followers count, is the duration taken to write 200 tweets related in any way to the followers count?

plt.scatter(x=df[q_50].followers_count,
            y=tweets_duration_org[q_50]);
plt.xscale('log')

# I can't see anything in this plot other than randomness, because of the outliers in tweets duration. Maybe if we were to look only at the users who tweeted close to 200 original tweets we'd get something.

plt.scatter(x=df[q_50][q_50_100].followers_count,
            y=tweets_duration_org[q_50][q_50_100]);
plt.xscale('log')

q_50_199 = df[q_50].tweets_count_org >= 199

# The randomnes still persists, but maybe if we just looked at duration between 0 and 500 we'd find something.

plt.scatter(x=df[q_50][q_50_100].followers_count,
            y=tweets_duration_org[q_50][q_50_100]);
plt.xscale('log')
plt.ylim([0, 500]);

# And the randomness still persists. What I noticed now and didn't notice before is that some people tweeted 200 tweets in just one day! How?

# ### Of the users who tweeted 200 tweets in just one day, is this their regular tweeting habits?

# To calculate this, we could look into the frequency of their tweets to get an estimate on the number of tweets per day that they tweet.

users_df['created_at'] = pd.to_datetime(users_df['created_at']).apply(lambda x: x.replace(tzinfo=None))

users_df['days'] = users_df.apply(lambda x: (collection_date - x.created_at).days, axis=1)


# calculate tweeting frequency account lifetime
# since some accounts were created the day of the collection, their days values will be 0
# and the frequency will be count/days, therefore we need to return just the count in case days is 0
def calculate_tweets_frequency(row):
    if row.days == 0:
        return row['statuses_count']
    else:
        return row['statuses_count'] / row['days']


users_df['tweets_per_day'] = users_df.apply(calculate_tweets_frequency, axis=1)

# only users for whom we have latest 200 tweets
tmp = users_df[q_50][tweets_duration[q_50] == 0]

bins = np.arange(0, 350, 50)
sns.histplot(tmp.tweets_per_day, bins=bins);

sns.histplot(users_df[q_50].tweets_per_day);

print('Median tweets per day for all users: {:.2f}'.format(users_df[q_50].tweets_per_day.median()))

# We can conclude that average tweets per day for these accounts isn't necessairly 200 tweets per day, but it's still significantly higher then average which is 3.74. What could be interesting is to actually look at how long they have been on twitter.
#
# But how much of this activity is actually original? Afterall, they could be just on twitter all day reading tweets and retweeting them. Could we tease out if these users are contributors or not?
#
# Well, the first thing to ask is are they popular?

bins = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
sns.histplot(tmp.followers_count, log_scale=True)
plt.xticks(bins, ['10', '100', '10K', '100K', '1M', '10M']);

# The distribution is bimodal, and what is interesting is that we have 12 users (out of 20) that exceed 100K followers. So the majority of these users have a lot of followers. But why do they stand out? Compared to the average time a users takes to tweet 200 tweets, these accounts take just one day. 
#
# How many of their tweets were original? How many were retweets? And how many were replies?
#
# Moreover do they get many reactions to their content? and exactly how many hours did it take them tweet these tweets? And how is the duration in hours related to other features like their followers count?

# I think that there is something imporant to do, and that's to cut the followers count into several categories in order to avoid using the log scale over and over.

bins = [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 200000000]
followers_labels = ['0-10', '10-100', '100-1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '>=10M']
users_df['followers_binned'] = pd.cut(users_df.followers_count, bins=bins, labels=followers_labels, include_lowest=True)

# Let's ask the previous question again, are they popular?

users_df.loc[:, old_feats+new_feats] = users_df.loc[:, old_feats+new_feats].fillna(0)
tmp = users_df[q_50][tweets_duration[q_50] == 0]

tmp.followers_binned.value_counts().sort_values(ascending=True).plot(kind='barh');

# Let's look at the statistics of each group of them to determine how does each followers category change their activity.

for label in tmp.followers_binned.unique():
    print(label, 'median original tweets {:.2f}'.format(
        tmp.query('followers_binned == @label').tweets_count_org.median()))
    print(label, 'median retweets tweets {:.2f}'.format(
        tmp.query('followers_binned == @label').tweets_count_rt.median()))
    print(label, 'median replies {:.2f}'.format(
        tmp.query('followers_binned == @label').tweets_count_rp.median()))
    print()

# We can see that users with 1M to 10M followers tend to have less retweets count than other categories, but I have to say that the sample size isn't large enought to actually look into these users' behavior. 

# Does these account get many reactions to their content?

for label in tmp.followers_binned.unique():
    print(label, 'median total retweets {:.2f}'.format(
        tmp.query('followers_binned == @label').total_retweets_org.median()))
    print(label, 'median total favorites {:.2f}'.format(
        tmp.query('followers_binned == @label').total_favorites_org.median()))
    print()

# It makes sense that users with huge following  have the highest retweets and favorites, but users in the 100k-1M category have values less than 1K-100K which is fishy, and that's because the sample isn't representative at all.
#
# Since there is no particular insight gained from these users since they are so few, we can look at their proportion of the sample we are using.

print('Proportion of users who tweeted 200 in one day is {:.2f}'.format(tmp.shape[0]/users_df[q_50].shape[0]))

# Which brings us to our next question.
# ## 2. What is the proportion of users who have tweeted more than 100 original tweets in the their last 200 tweets?
#

users_df[q_50][q_50_100].shape[0] / users_df[q_50].shape[0]

# 43% of users have tweeted more than 100 original tweets in their latest 200 tweets.

# Another question would be 
# ## What is the proportion of users who have retweeted more than 100 tweets of their last 200?

users_df[q_50].query('tweets_count_rt >= 100').shape[0] / users_df[q_50].shape[0]

# 11% of users have retweeted more than 100 tweets of their latest 200 tweets.

# ## 3. What are the characterstics of users who don't retweet and those who do?
#
# First, how do we define a user who retweets from a user who doesn't retweet? Let's take a look at the distribution of the retweets count.

bs = 20
bins = np.arange(0, 200+bs, bs)
sns.histplot(users_df[q_50].tweets_count_rt, bins=bins)
plt.xticks(bins);

# We can make several classes based on 5 numbers summary that best describe the users retweeting habits, and I could make another cut for these bins based on my intuition.

users_df[q_50].tweets_count_rt.describe()

bins = [0, 6, 21, 56, 200]
# bins = [0, 25, 50, 100, 200]
retweet_labels = ['Barely retweets', 'Occasionaly retweets', 'Moderatly retweets', 'Always retweets']
users_df['retweet_class'], bins = pd.cut(users_df.tweets_count_rt, bins=4, labels=retweet_labels, retbins=True, right=False)

(users_df[q_50]['retweet_class'].value_counts()/users_df[q_50].shape[0]).plot(kind='bar');

# 70% of users retweeted less than 25% of their last 200 tweets, which indicates that this is the more general behavior on twitter, meanwhile the remanining 30% retweeted more than 25%.

# ### How long has these users been using twitter?

plt.figure(figsize=(10, 5))
sns.histplot(data=users_df[q_50], x='days', hue='retweet_class', element='step')

g = sns.catplot(data=users_df[q_50], x='days', y='retweet_class', kind='violin');

# There is no particular insight in this plot
#
# ### How many followers do they typically have?

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
fig.tight_layout(pad=5.0)
axes = axes.flatten()
for i, label in enumerate(retweet_labels):
    group = users_df[q_50].query('retweet_class == @label')
    followers_binned = group.followers_binned.value_counts()[followers_labels]
    followers_binned_prop = followers_binned/group.shape[0]
    followers_binned_prop.plot(kind='bar', ax=axes[i])
    axes[i].set_title(label)
    
    for j, v in enumerate(followers_binned_prop.tolist()):
        axes[i].text(j-0.25, v+0.01, str(followers_binned[j]))


# Since the majority of users barely retweet, we can see a somewhat equal representation of different user groups in terms of followers count. On the other hand, each of the 2 following retweeting behaviors show somewhat similar pattern, 

plt.figure(figsize=(15, 10))
ticks = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
ticklabels = ['10', '100', '1K', '10K', '100K', '1M', '10M', '100M']
for i, label in enumerate(labels):
    plt.subplot(2, 2, i+1)
    sns.histplot(users_df[q_50].query('retweet_class == @label').followers_count+0.000001, log_scale=True)
    plt.title(label)
    plt.xticks(ticks, ticklabels);

# We can see that the majority of million followers club of twitter is present in the first two categories, indicating that most of these users tend to retweet less often.
#
# ### How much attention do they typically get?

plt.figure(figsize=(15, 10))
# ticks = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
# ticklabels = ['10', '100', '1K', '10K', '100K', '1M', '10M']
for i, label in enumerate(labels):
    plt.subplot(2, 2, i+1)
    sns.histplot(users_df[q_50].query('retweet_class == @label').avg_retweets_org)
#     plt.title(label)
#     plt.xticks(ticks, ticklabels);

# ### Using ordinal data will help illuminate any patterns in this, so I think that I'd better cut values of duration of lifetime and follower, etc.. before making any new plots and make the older plots accordingly.

# ## To be continued...

# ## Taking a broad look at the features

num_feats = ['friends_count', 'followers_count', 'statuses_count', 'favourites_count', 'months']
sns.pairplot(data=users_df, vars=num_feats, plot_kws={'alpha': 0.5, 'bins': 100});

# This plot shows that in each these features, except months since creation, there are outliers that make the range of the plots larger than the actual spread of the data. Also the histograms of the 4 features are barely visible also due to the presence of outliers.

# ## Analyzing users' friends and followers distributions

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

users_df.friends_count.plot(kind='hist', ax=axes[0], bins=100, title='Friends Count Distribution')
users_df.followers_count.plot(kind='hist', ax=axes[1], bins=100, title='Followers Count Distribution');
# -

# From the get go we can see that there alot of outliers in both the friends and followers count features, and the distributions look to be right skewed.

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

users_df.friends_count.plot(kind='box', ax=axes[0], title='Friends Count Boxplot')
users_df.followers_count.plot(kind='box', ax=axes[1], title='Followers Count Bocplot');
# -

# These two distributions look log normal to me, so let's look at their log transformation.

# +
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=users_df.query('friends_count > 0'), x='friends_count', log_scale=True);
plt.title('Log Transformation of Friends Count')

plt.subplot(1, 2, 2)
sns.histplot(data=users_df.query('followers_count > 0'), x='followers_count', log_scale=True)
plt.title('Log Transformation of Followers Count');

# -

users_df[['friends_count', 'followers_count']].describe()

# The two distributions are log normal, and if we look at the statistical summary of the two features, we find that 50th qunatile in friends is 433, which means that half of the users in this data have friends less than or equal to 433, while for followers it is 658.

print('Number of users with 0 friends:', users_df.query('friends_count == 0').shape[0])
print('Number of users with 0 followers:', users_df.query('followers_count == 0').shape[0])
print('\nMaximum of number of friends:', users_df.friends_count.max())
print('Maximum of number of followers:', users_df.followers_count.max())

# Why would someone have no friends? or followers? Are these new users?
# Also having a lot of followers is not surprising, even they are 20 million, but why would someone have 79k friends?

# ### Analyzing users with no friends

sns.pairplot(data=users_df.query('friends_count == 0'), vars=num_feats, height=1.5);

# I don't think there is anything interesting in these plots except one fact, and that is how their followers count is distributed.

sns.histplot(data=users_df.query('friends_count == 0'), x='followers_count', log_scale=True);

users_df.query('friends_count == 0')[num_feats].describe()

# Based on the plots and statistical description of these users, it's apparent the majority of them aren't inactive or new accounts. 75% of them have been present on twitter for more than two years, also 75% of them have more than 1988 followers and 142 statuses.
#
# But we can see that 25% of them have no favorites, so these might include news or business accounts.

print('Number of users without friends or favorites:', 
       users_df.query('friends_count == 0 and favourites_count == 0').shape[0])
print('Number of users without friends that haven\'t completed on month since creation:', 
       users_df.query('friends_count == 0 and months == 0').shape[0])

sns.histplot(users_df.query('friends_count == 0 and favourites_count == 0'), x='followers_count', log_scale=True);

sns.histplot(users_df.query('friends_count == 0 and favourites_count == 0'), x='statuses_count', log_scale=True);

sns.scatterplot(data=users_df.query('friends_count == 0\
                                     and favourites_count == 0'), x='months', y='statuses_count');

tmp = users_df.query('friends_count == 0 and favourites_count == 0')
plt.scatter(x=tmp['months'],
            y=tmp['statuses_count'],
            c=tmp['followers_count'])

# We can see that there is no direct relationship between these features, so let's now take a look at the accounts to figure out exactly what they are.

users_df.query('friends_count == 0').head(50)

# +
# # how many duplicates do we have?
# print('Number of duplicated rows {0}'.format(users_df[users_df['id'].duplicated()].shape[0]))

# +
# # drop duplicates
# users_df = users_df[~users_df['id'].duplicated()]
# -

# NaN per column
users_df.isna().sum(axis=0)


# +
# fill out the last tweets count by the difference between the total distinct tweets in db and sum of dataframe
# users_df.loc[users_df.tweets_count.isna(), 'tweets_count'] = len(db.tweets.distinct('id')) -\
#                                                              users_df.tweets_count.sum().astype(int)

# +
# users_df.tweets_count.sum()

# +
# len(db.tweets.distinct('id'))

# +
# users_df.tweets_count.sum() - len(db.tweets.distinct('id'))

# +
# users_df.query('tweets_count > 200')[['id', 'tweets_count']]
# -

def collect_tweets_count(user_id):
    count = len(db.tweets.find({'user.id': user_id}).distinct('id'))
    return count


# +
## used to inspect the problem of different tweets count between df and collection

# users_tweets_count = []
# users = users_df.query('tweets_count > 200').id
# tweets_count = users_df.query('tweets_count > 200').tweets_count
# for user_id, user_count in zip(users, tweets_count):
#     user_tweets_count = collect_tweets_count(user_id)
#     print('{}, {}, {}'.format(user_id, user_count, user_tweets_count))
#     users_tweets_count.append(user_tweets_count)
    
# print('Difference between old and new tweets count: {}'.format(tweets_count.sum() - sum(users_tweets_count)))

# +
# users_tweets_count = []
# sample = users_df.sample(20)
# users = sample.id
# tweets_count = sample.tweets_count
# for user_id, user_count in zip(users, tweets_count):
#     user_tweets_count = collect_tweets_count(user_id)
#     print('{}, {}, {}'.format(user_id, user_count, user_tweets_count))
#     users_tweets_count.append(user_tweets_count)
    
# print('Difference between old and new tweets count: {}'.format(tweets_count.sum() - sum(users_tweets_count)))

# +
# # aggregate users tweets count, average retweet and favorite count, and total retweet and favorite count
# results = db.tweets.aggregate([
#     {
#         '$match': {'user.id': 474886468}
#     },
#     {
#         '$group':{
#             '_id': {
#                 'user_id': '$user.id',
#                 'screen_name': '$user.screen_name',
#                 'tweet_id': '$id',

#             },
#             'count': {'$sum': 1},
#             'retweet_count': {'$max': '$retweet_count'},
#             'favorite_count': {'$max': '$favorite_count'}
#         }
#     },
#     {
#         '$group':{
#             '_id': {
#                 'user_id':'$_id.user_id',
#                 'screen_name': '$_id.screen_name'
#             },
#             'count': {'$sum': 1},
#             'avg_rt': {'$avg': '$retweet_count'},
#             'avg_fv': {'$avg': '$favorite_count'},
#             'total_rt': {'$sum': '$retweet_count'},
#             'total_fv': {'$sum': '$favorite_count'},
#         }
#     }
#     ], allowDiskUse=True)

# ids = []
# for i in results: 
# #     if i['_id']['tweet_id'] in ids:
# #         print('DUPLICATE', i)
#     print(i)
# #     ids.append(i['_id']['tweet_id'])
# # print(len(ids), len(set(ids)))
# -

# I figured out the problem, it is that the data was collected in multiple times, and several users were duplicated with their tweets, however their tweets had different number of favorites and retweets now, and since I was using retweet and favorite counts as ids, duplicates of tweets weren't duplicates anymore.
#
# So now instead of taking retweet and favorite count as ids, I will aggregate the max of their duplicates in order to get the latest information.

# +
# users_df.to_csv('users.csv', index=False)
# -

# ## Analyzing newly added features before dropping outliers

cols = ['statuses_count', 'favourites_count', 'followers_count', 'friends_count']
new_cols = users_df.iloc[:, -5:].columns.tolist()

users_df[new_cols].hist(figsize=(12, 8), bins=50);

users_df[new_cols].plot(kind='box', subplots=True, figsize=(16, 5));

# We can see that the presence of outliers won't enable us to see to any meaningful chart of the last 4 features.

# +
# heatmap

sns.heatmap(users_df[new_cols].corr(), annot=True);
# -

# There are obvious relationships between average features derived from total ones, but there are still relationships between the average and the average favorites of a user.

plt.figure(figsize=(10, 6))
sns.heatmap(users_df[cols+new_cols].corr(), annot=True);

# The two histograms have outliers (of course), so let's boxplot them to check them out.

users_df.loc[:, ['friends_count', 'followers_count']].describe()

# Let's take a look at a pair plot between all features to find out if there are any pecularities.

# +
# sns.pairplot(data=users_df, vars=cols, plot_kws={'alpha': 0.2});
# -

# The outliers won't enable any real insight to be taken from the distribution of users, so let's take a look at a clean version of the data.

# +
thresh = 0.99
# outliers_mask = (users_df.friends_count < users_df.friends_count.quantile(thresh)) &\
#                 (users_df.followers_count < users_df.followers_count.quantile(thresh))

# outliers_mask = users_df.followers_count < users_df.followers_count.quantile(thresh)

# +
outliers_mask = (users_df.friends_count > users_df.friends_count.quantile(thresh)) &\
                (users_df.followers_count > users_df.followers_count.quantile(thresh))

outliers_mask.sum()

# +
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=users_df[outliers_mask].query('friends_count > 0'), x='friends_count', log_scale=True);
plt.title('Log Transformation of Friends Count')

plt.subplot(1, 2, 2)
sns.histplot(data=users_df[outliers_mask].query('followers_count > 0'), x='followers_count', log_scale=True)
plt.title('Log Transformation of Followers Count');
# -

# By clipping the distributions after the 99th quantile, we can see that the distribution of the log transformation is normal, but how does this affect the results of further analysis? Let's formalize these questions:
#
# 1. What is the effect of leaving the outliers on further analysis?
# 2. What is the effect of removing the outliers on further analysis?
#
# To answer these question, I need to take a closer look at these outliers to understand
# 1. Are they essential to the data understanding?
# 2. Will removing them affect my attempt in clustering the data?

# +
# take a look at outliers with high followers count
thresh = 0.95
outliers_mask = users_df.followers_count < users_df.followers_count.quantile(thresh)

print('The number of outliers the fit this criteria is {0}'.format(users_df[~outliers_mask].shape[0]))

# +
# get to know the users 
users_df[~outliers_mask].sort_values('followers_count', ascending=False)

# for i, row in  users_df[~outliers_mask].sort_values('followers_count').iterrows():
#     print(row['name'], row['followers_count'], row['statuses_count'])
# -

# It is obvious that these users are either celeberties, official and non-official accounts for celeberities, entities, new outlets and twitter pages of sorts. So They aren't regular twitter users or influencers.

# the distribution of their numeric features
axes = users_df[~outliers_mask][cols].hist(bins=10, figsize=(15, 10))
users_df[~outliers_mask][cols].describe()

# +
# take a look at outliers with high friends count
thresh = 0.95
outliers_mask = users_df.friends_count < users_df.friends_count.quantile(thresh)

print('The number of outliers the fit this criteria is {0}'.format(users_df[~outliers_mask].shape[0]))
# -

# get to know the users 
users_df[~outliers_mask].sort_values('friends_count', ascending=False).tail(50)

# the distribution of their numeric features
axes = users_df[~outliers_mask][cols].hist(bins=10, figsize=(15, 10))
users_df[~outliers_mask][cols].describe()

# I have tried clipping values beyond the 95th and 99th percentile, and I'll go with the more conservative 99th, as I don't want to lose much of the data. Clipping beyond 95th removes a lot of rubbish, but I think it can remove valuable data points.
#
# I'll reflect on this decision when I run the Gaussian Mixture Clustering again.

# +
thresh = 0.95

# friends_count > 99th quantile OR followers_count > 99th quantile
outliers_mask = (~((users_df.friends_count > users_df.friends_count.quantile(thresh)) |
                 (users_df.followers_count > users_df.followers_count.quantile(thresh))))

print('Rows removed: {0}'.format(users_df[~outliers_mask].shape[0]))

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

users_df[outliers_mask].friends_count.plot(kind='box', ax=axes[0], title='Friends Count Boxplot')
users_df[outliers_mask].followers_count.plot(kind='box', ax=axes[1], title='Followers Count Boxplot');
# -

users_df.loc[outliers_mask, ['friends_count', 'followers_count']].describe()

# I'll use the median as a measure of centre for the rest of the analysis, as the data is still heavilt right skewed.

print('Number of outliers:', users_df.shape[0] - users_df[outliers_mask].shape[0])
print('Outliers friends and followers summary:')
users_df.loc[~outliers_mask, ['friends_count', 'followers_count']].describe()

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

users_df[~outliers_mask].friends_count.plot(kind='hist', ax=axes[0], bins=50, title='Friends Count Distribution')
users_df[~outliers_mask].followers_count.plot(kind='hist', ax=axes[1], bins=50, title='Followers Count Distribution');

# +
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=users_df[outliers_mask].query('friends_count > 0'), x='friends_count', log_scale=True);
plt.title('(Log Transformed) Friends Count Distribution')

plt.subplot(1, 2, 2)
sns.histplot(data=users_df[outliers_mask].query('followers_count > 0'), x='followers_count', log_scale=True)
plt.title('(Log Transformed) Followers Count Distribution');
# -

users_df[~outliers_mask].plot(kind='scatter',
                              x='friends_count',
                              y='followers_count', 
                              alpha=0.2,
                              figsize=(10, 6));

# These outliers are distributed on the two axes, so there are basically people who have really low followers count and huge friends count or the opposite. Let's take a look at the cleaned version.

users_df[outliers_mask].plot(kind='scatter',
                             x='friends_count',
                             y='followers_count',
                             figsize=(10, 6),
                             alpha=0.2);

# In the previous version, this scatter plot was a little bit different, as the maximum friends count was 9k, but now it is 69k.
#
# Previous version analysis:
#
#     There is some sort of linear relationship going on below 25000 followers. I guess that these are users that get followers by following other users in return of these users following them back. That's why these have an apparent linear relationship between friends and followers count. On the other hand we can see the other distribution of users who don't show a linear relationship between these two features, and these may be the users who have amassed a following based on their activity, and not relying on quid pro quo agreement with other users to follow them and expect a follow back.

users_df[outliers_mask][new_cols].hist(figsize=(12, 8), bins=50);

users_df[outliers_mask][new_cols].plot(kind='box', subplots=True, figsize=(16, 5));

# It is obvious now that it's not a problem of outliers for visualization with this data, it's that the data is heavily skewed. Therefore I think that the only metric for deciding whether to remove outliers or not is to examine the results of clustering the data.

# +
import pytz
import datetime
from dateutil.relativedelta import relativedelta

utc=pytz.UTC

now = utc.localize(datetime.datetime.now())

users_df['delta'] = pd.to_datetime(users_df.created_at).apply(lambda x: relativedelta(now, x))
users_df['years'] = pd.to_datetime(users_df.created_at).apply(lambda x: relativedelta(now, x).years)
users_df['months'] = users_df.delta.apply(lambda x: x.years * 12 + x.months)
users_df['days'] = users_df.delta.apply(lambda x: x.years * 365 + x.days)

cols = cols + new_cols + ['months']


# -

def scatter_cluster(tmp, size_col='statuses_count', scale=1):
    plt.figure(figsize=(10, 6))
    plt.scatter(x=tmp['friends_count'],
                y=tmp['followers_count'],
                s=tmp[size_col]/scale,
                c=tmp['months'])
    plt.colorbar()


scatter_cluster(users_df[outliers_mask], 'avg_favorites', scale=10)

# Based on this plot and changing the size col parameters with the new engineered features, I believe that the algorithm will be able to capture new and better information.

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(data=users_df[outliers_mask], x='total_retweets', y='total_favorites', ax=axes[0]);
sns.scatterplot(data=users_df[outliers_mask], x='avg_retweets', y='avg_favorites', ax=axes[1]);

# We can see the positive correlation between the total and average number of retweets and favorites between users.

sns.histplot(data=users_df[outliers_mask].query('total_retweets > 0'), x='total_retweets', log_scale=True);

sns.histplot(data=users_df[outliers_mask].query('total_favorites > 0'), x='total_favorites', log_scale=True);


def plot_hist(feature, mask_outliers=True, log_scale=False):
    if log_scale:
        mask = users_df[feature] > 0
        if mask_outliers:
            sns.histplot(data=users_df[outliers_mask][mask], x=feature, log_scale=True)
        else:
            sns.histplot(data=users_df[mask], x=feature, log_scale=True)
    else:
        if mask_outliers:
            sns.histplot(data=users_df[outliers_mask], x=feature)
        else:
            sns.histplot(data=users_df, x=feature)


plot_hist('avg_retweets', mask_outliers=False, log_scale=True);

plot_hist('avg_retweets', mask_outliers=True, log_scale=True);

plot_hist('avg_favorites', mask_outliers=False, log_scale=True);

plot_hist('avg_favorites', mask_outliers=True, log_scale=True);

# The increased bin size near 1 average favorites is peculiar, why does removing outliers decrease this bin size?

users_df[~outliers_mask].avg_favorites.hist(bins=50);

# I guess that these outliers are people with huge number of friends, let's check this out.

users_df[~outliers_mask].plot(kind='scatter', x='friends_count', y='avg_favorites');
plt.ylim([0, 100]);

# That turned out to be true, and most this bin is mostly inhabited by outliers with huge friends count.

plt.figure(figsize=(10, 6))
plt.scatter(x=users_df[outliers_mask]['avg_retweets'],
            y=users_df[outliers_mask]['avg_favorites'],
            alpha=0.2,
            c=users_df[outliers_mask]['followers_count'],
            s=users_df[outliers_mask]['months']);
plt.colorbar()
plt.xlim([0, 1])
plt.ylim([0, 10]);

# Zooming in on the data shows that new users and are one the are centered around the origin.

plt.figure(figsize=(10, 6))
plt.scatter(x=users_df[outliers_mask]['total_retweets'],
            y=users_df[outliers_mask]['total_favorites'],
            alpha=0.2,
            c=users_df[outliers_mask]['followers_count'],
            s=users_df[outliers_mask]['months']);
plt.colorbar();

plt.figure(figsize=(10, 6))
plt.scatter(x=users_df[outliers_mask]['statuses_count'],
            y=users_df[outliers_mask]['avg_retweets']);

# No relationship is present between the numbers of statuses a user has and the average number of retweets he has.

plt.figure(figsize=(10, 6))
plt.scatter(x=users_df[outliers_mask]['statuses_count'],
            y=users_df[outliers_mask]['avg_favorites']);

plt.figure(figsize=(10, 6))
plt.scatter(x=users_df[outliers_mask]['tweets_count'],
            y=users_df[outliers_mask]['avg_favorites']);

scatter_cluster(users_df[outliers_mask], 'tweets_count', scale=1)

# It seems that since the latest 200 tweets are collected over any time, whether recent or distant, therefore I need to engineer a feature with the average number of tweets per month for example.

cols

# I'll now make up different categories based on followers to analyze the data.

# ## Clustering users based on friends and followers count
#
# I have removed all previous attempts to cluster using kmeans as they were subpar compared to gaussian mixture models.

# +
# Now it's time to save these results to enable further analysis based on them
import os
import sys
import pickle as pk

from datetime import datetime


def save_model(model, filename, save_dir='models'):
    filename = '{}_{}'.format(filename, datetime.now())
    filepath = os.path.join(save_dir, filename)
    file = open(filepath, 'wb')
    
    pk.dump(model, file)
    print('Model saved as {} at {}/'.format(filename, save_dir) , file=sys.stderr)
    
def load_model(filename, save_dir='models'):
    filepath = os.path.join(save_dir, filename)
    file = open(filepath+'.pkl', 'rb')
    
    model = pk.load(file)
    print('{} loaded from {}'.format(filename, save_dir), file=sys.stderr)
    return model


# -

users_df[old_feats]

# +
from sklearn.mixture import GaussianMixture

n = 2
gm = GaussianMixture(n_components=n, n_init=20, random_state=seed)
gm.fit(users_df[old_feats+new_feats])

# gm = load_model('gm_95th_2021-04-08 13:20:35.718419.pkl')

y_pred_4 = gm.predict(users_df[old_feats+new_feats])

plt.figure(figsize=(10, 6))

for label, color in zip(set(y_pred_4), 'bgrcmy'):
    plt.scatter(x=users_df[y_pred_4 == label]['friends_count'],
                y=users_df[y_pred_4 == label]['followers_count'],
                c=color,
                alpha=0.5,
                label=label);

plt.legend();

# +
plt.figure(figsize=(10, 6))
for label, color in zip(set(y_pred_4), 'bgrcmy'):
    plt.scatter(x=users_df[y_pred_4 == label]['friends_count'],
                y=users_df[y_pred_4 == label]['followers_count'],
                c=color,
                s=users_df[y_pred_4 == label]['avg_retweets_org']/1,
                alpha=0.5,
                label=label);

plt.legend();
# -

# The results are exceptional and by analyzing these groups, we might be able to extract new insights based on the users data.

gm_0 = users_df[outliers_mask][y_pred_4 == 0]
gm_1 = users_df[outliers_mask][y_pred_4 == 1]
gm_2 = users_df[outliers_mask][y_pred_4 == 2]
gm_3 = users_df[outliers_mask][y_pred_4 == 3]
gm_4 = users_df[outliers_mask][y_pred_4 == 4]
gm_5 = users_df[outliers_mask][y_pred_4 == 5]

scatter_cluster(gm_0, scale=100)
gm_0[cols].describe()

sns.pairplot(data=gm_0, vars=cols, height=1.5, kind='scatter', plot_kws={'alpha': 0.1});

gm_0.head(50).sort_values('followers_count', ascending=False)[['screen_name']+cols]

sns.scatterplot(data=gm_0, x='friends_count', y='followers_count', alpha=0.2);

sns.histplot(data=gm_0.query('friends_count > 0'), x='friends_count', log_scale=False);

sns.histplot(data=gm_0.query('followers_count > 0'), x='followers_count', log_scale=False);

sns.scatterplot(data=gm_0, x='friends_count', y='favourites_count', alpha=0.2);

sns.scatterplot(data=gm_0, x='friends_count', y='statuses_count', alpha=0.2);

# Judging by the average friends count which is higher than followers count, and the average number of statuses and favourites compared to the number of months, these are the new and inactive users. The thing that brings this cluster together is the diminishing activity, whether in terms of original tweets, retweets or favourites.
#
# We can notice the spread of the distribution in friends count is more than followers count.
#
# We can also see along the the y axis, when friends count is near to zero, some profiles that have nearly now friends but have near thousands of followers. These are definetly pages.
#
# The way the this data was collected could enable more insight into this group, as the latest original 200 tweets (if they existed) were collected for every user. So if any of these users didn't have one original tweet or reply in the latest 200 tweets, then we can judge that they aren't active at all.

# +
# # %%time
# db.tweets.count_documents({'user.screen_name': row.screen_name})

# +
# result = db.tweets.aggregate([
#     {
#          '$match':
#          {
#              'user.screen_name': 'MSha3bo'
#          }        
#     },
#     {
#         '$unwind': '$user.screen_name'
#     },
#     {
#         '$lookup': {
#             'from': 'users',
#             'localField': 'user.screen_name',
#             'foreignField': 'screen_name',
#             'as': 'data'
#         }
#     },
#     {
#         '$group': {
#             '_id': '$data.id',
#             'count': {'$sum': 1}
#         }
#     }
# ])

# for i in result:
#     print(i)

# +
# result = db.tweets.aggregate(
#     [    {
#              '$match':
#              {
#                  'user.screen_name': 'MSha3bo'
#              }
            
#          },
#          {
#              '$lookup':
#               {
#                 'from': 'users',
#                 'localField': 'user.screen_name',
#                 'foreignField': 'screen_name',
#                 'as': 'user_tweets'
#               }
#          },
#          {
#              '$group':
#               {
#                   '_id': '$user_tweets.screen_name',
#                   'count': {'$sum': 1}
#               }
#          }
#     ]
# )

# for i in result:
#     print(i)

# +
# tmp = pd.DataFrame(list(db.tweets.find({'user.screen_name': 'MSha3bo'})))
# tmp.head()

# +
# tmp[~tmp.id.duplicated()]

# +
# tweets_count = []
# i = 0
# for _, row in gm_0.iterrows():
#     tweets_count.append(collect_tweets_counts(row.screen_name))
#     i += 1
#     print('\r{}/{}'.format(i, len(gm_0)), end='')
# gm_0['tweets_count'] = tweets_count
# -

sns.histplot(data=gm_0, x='tweets_count');

# The assumption that I've made about the data was true, these are users who mostly don't even tweet.

sns.scatterplot(data=gm_0, x='tweets_count', y='followers_count');

scatter_cluster(gm_1, 'avg_retweets', scale=1)
gm_1[cols].describe()

gm_1.head(50)

sns.pairplot(data=gm_1, vars=cols, height=1.5, plot_kws={'alpha': 0.2});

# Users who aren't really active, but they are more active than the first cluster.

scatter_cluster(gm_2, 'avg_retweets', scale=1)
gm_2[cols].describe()

gm_2

sns.pairplot(data=gm_2, vars=cols, height=1.5, plot_kws={'alpha': 0.2});

print('Percentage of users who tweet in cluster 2: {:.2f}%'.format(gm_2.query('tweets').size*100/ gm_2.size))
print('Percentage of users who don\'t tweet in cluster 2: {:.2f}%'.format(gm_2.query('~tweets').size*100/ gm_2.size))

users_df.query('tweets').size*100/users_df.size

sns.histplot(data=users_df, x='tweets_count');

# Influencers

scatter_cluster(gm_3, 'avg_favorites', scale=1)
gm_3[cols].describe()

gm_3[['screen_name']+cols].head(50)

sns.pairplot(gm_3, vars=cols, height=1.5, plot_kws={'alpha': 0.2});

# Mostly official and non-official pages that represents entities or personalities.

scatter_cluster(gm_4, 'avg_retweets', scale=1)
gm_4[cols].describe()

gm_4.head(50).sort_values('statuses_count')[['name']+cols]

sns.pairplot(gm_4, vars=cols, height=1.5, plot_kws={'alpha': 0.2});

# This cluster might have some high followers, but it shouldn't fool you as most of them got it using follow backs. You can also see it from the low ratio between favourites and statuses count. You can also see some users with really low statuses count, but their followers is high.

scatter_cluster(gm_5, 'avg_retweets', scale=1)
gm_5[cols].describe()

gm_5.sort_values('followers_count').head(50)

sns.pairplot(gm_5, vars=cols, height=1.5, plot_kws={'alpha': 0.2});

# +
# save_model(gm, 'gm_95th')
# -

gm__2 = load_model('gm_2021-04-07 12:55:26.141024.pkl')

# # Clustering using PCA and KMeans
#
#
# First let's reduce the data using PCA and visualize it 

# +
from sklearn.decomposition import PCA

pca = PCA()
users_2d = pca.fit_transform(users_df[outliers_mask][cols])
# -

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

plt.scatter(users_2d[:, 1], users_2d[:, 2])

# +
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K = [3, 4, 5, 6]
scores = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(users_2d)
    scores.append(silhouette_score(users_2d, kmeans.labels_))
    
plt.plot(K, scores, '-o');
# -

k = 4  # highest score
kmeans = KMeans(n_clusters=k)
kmeans.fit(users_2d)

km_0 = users_df[outliers_mask][kmeans.labels_ == 0]
km_1 = users_df[outliers_mask][kmeans.labels_ == 1]
km_2 = users_df[outliers_mask][kmeans.labels_ == 2]
km_3 = users_df[outliers_mask][kmeans.labels_ == 3]

scatter_cluster(km_0, scale=100)
km_0[cols].describe()

scatter_cluster(km_1, scale=100)
km_1[cols].describe()

scatter_cluster(km_2, scale=100)
km_2[cols].describe()

scatter_cluster(km_3, scale=100)
km_3[cols].describe()

# The results of kmeans aren't nearly as good as the gaussian mixture.

# engineered features
users_df['followers_friends_ratio'] = users_df['followers_count'] / (users_df['friends_count'] + 0.00001)
users_df['statuses_favourites_ratio'] = users_df['statuses_count'] / (users_df['favourites_count'] + 0.00001)
users_df['statuses_per_month'] = users_df['statuses_count'] / (users_df['months'] + 0.00001)
users_df['favourites_per_month'] = users_df['favourites_count'] / (users_df['months'] + 0.00001)

# +
n = 10

cols2 = cols + ['followers_friends_ratio', 'statuses_favourites_ratio',
                'statuses_per_month', 'favourites_per_month']

gm2 = GaussianMixture(n_components=n, n_init=20, random_state=92)
gm2.fit(users_df[outliers_mask][cols2])

y_pred_5 = gm2.predict(users_df.loc[outliers_mask, cols2])

plt.figure(figsize=(10, 6))

for label, color in zip(set(y_pred_5), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred_5 == label]['friends_count'],
                y=users_df[outliers_mask][y_pred_5 == label]['followers_count'],
                c=color,
                alpha=0.5,
                label=label);

plt.legend();

# +
plt.figure(figsize=(10, 6))
for label, color in zip(set(y_pred_5), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred_5 == label]['friends_count'],
                y=users_df[outliers_mask][y_pred_5 == label]['followers_count'],
                c=color,
                s=users_df[outliers_mask][y_pred_5 == label]['statuses_count']/100,
                alpha=0.5,
                label=label);

plt.legend();
# -

gm2_0 = users_df[outliers_mask][y_pred_5 == 0]
gm2_1 = users_df[outliers_mask][y_pred_5 == 1]
gm2_2 = users_df[outliers_mask][y_pred_5 == 2]
gm2_3 = users_df[outliers_mask][y_pred_5 == 3]
gm2_4 = users_df[outliers_mask][y_pred_5 == 4]
gm2_5 = users_df[outliers_mask][y_pred_5 == 5]

scatter_cluster(gm2_0, scale=100)
gm2_0[cols].describe()

scatter_cluster(gm2_1, scale=100)
gm2_1[cols].describe()

scatter_cluster(gm2_2, scale=100)
gm2_2[cols].describe()

scatter_cluster(gm2_3, scale=100)
gm2_3[cols].describe()

scatter_cluster(gm2_4, scale=100)
gm2_4[cols].describe()

scatter_cluster(gm2_5, scale=100)
gm2_4[cols].describe()

# The results were poor compared to the old gaussian mixture model.

#
