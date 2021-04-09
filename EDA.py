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

# +
import pymongo
from pprint import pprint
from twitter_api import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

seed = 44
np.random.seed(seed)
# -

client = pymongo.MongoClient('localhost', 27017)
client.list_database_names()

db = client['twitter_stream']
db.list_collection_names()

db.tweets.find_one()

# querying tweets for certain user
count = 0
limit = 10
for tweet in db.tweets.find({'user.screen_name': 'MSha3bo'}):
    if count > limit:
        break
    pprint(tweet['full_text'])
    count += 1

# querying one user
db.users.find_one()

# +
# querying all screen_names
screen_names = []
for user in db.users.find():
    screen_names.append(user['screen_name'])
    
screen_names = set(screen_names)
print('Total number of users:', len(screen_names))

# +
# converting users collection into pandas dataframe
import pandas as pd

users_df = pd.DataFrame(list(db.users.find()))
users_df.head()
# -

# how many duplicates do we have?
print('Number of duplicated rows {0}'.format(users_df[users_df['id'].duplicated()].shape[0]))

# drop duplicates
users_df = users_df[~users_df['id'].duplicated()]

# NaN per column
users_df.isna().sum(axis=0)

# ## Analyzing users' friends and followers distributions

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

users_df.friends_count.plot(kind='hist', ax=axes[0], bins=100, title='Friends Count Distribution')
users_df.followers_count.plot(kind='hist', ax=axes[1], bins=100, title='Followers Count Distribution');
# -

# The two histograms have outliers (of course), so let's boxplot them to check them out.

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

users_df.friends_count.plot(kind='box', ax=axes[0], title='Friends Count Boxplot')
users_df.followers_count.plot(kind='box', ax=axes[1], title='Followers Count Bocplot');
# -

users_df.loc[:, ['friends_count', 'followers_count']].describe()

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
cols = ['statuses_count', 'favourites_count', 'followers_count', 'friends_count']
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
cols = ['statuses_count', 'favourites_count', 'followers_count', 'friends_count']
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

cols = ['friends_count', 'followers_count', 'statuses_count', 'favourites_count', 'months']


# -

def scatter_cluster(tmp, size_col='statuses_count', scale=1):
    plt.figure(figsize=(10, 6))
    plt.scatter(x=tmp['friends_count'],
                y=tmp['followers_count'],
                s=tmp[size_col]/scale,
                c=tmp['months'])
    plt.colorbar()


scatter_cluster(users_df[outliers_mask], 'statuses_count', scale=1000)

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


# +
from sklearn.mixture import GaussianMixture

# n = 10
# gm = GaussianMixture(n_components=n, n_init=20, random_state=seed)
# gm.fit(users_df[outliers_mask][cols])

gm = load_model('gm_95th_2021-04-08 13:20:35.718419.pkl')

y_pred_4 = gm.predict(users_df.loc[outliers_mask, cols])

plt.figure(figsize=(10, 6))

for label, color in zip(set(y_pred_4), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred_4 == label]['friends_count'],
                y=users_df[outliers_mask][y_pred_4 == label]['followers_count'],
                c=color,
                alpha=0.5,
                label=label);

plt.legend();

# +
plt.figure(figsize=(10, 6))
for label, color in zip(set(y_pred_4), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred_4 == label]['friends_count'],
                y=users_df[outliers_mask][y_pred_4 == label]['followers_count'],
                c=color,
                s=users_df[outliers_mask][y_pred_4 == label]['statuses_count']/100,
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

sns.pairplot(gm_0[cols], height=1.5);

gm_0.head(50)

# Inactive and new users

scatter_cluster(gm_1, scale=100)
gm_1[cols].describe()

sns.pairplot(gm_1[cols], height=1.5);

# Users who aren't really active.

scatter_cluster(gm_2, scale=100)
gm_2[cols].describe()

gm_2.head(50)

sns.pairplot(gm_2[cols], height=1.5);

# Influencers

scatter_cluster(gm_3, scale=100)
gm_3[cols].describe()

gm_3

sns.pairplot(gm_3[cols], height=1.5);

# Mostly official and non-official pages that represents entities or personalities.

scatter_cluster(gm_4, scale=100)
gm_4[cols].describe()

gm_4.head(50).sort_values('statuses_count')[['name']+cols]

sns.pairplot(gm_4[cols], height=1.5);

# This cluster might have some high followers, but it shouldn't fool you as most of them got it using follow backs. You can also see it from the low ratio between favourites and statuses count. You can also see some users with really low statuses count, but their followers is high.

scatter_cluster(gm_5, scale=100)
gm_5[cols].describe()

gm_5.sort_values('followers_count').head(50)

sns.pairplot(gm_5[cols], height=1.5);

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
users_df['followers_friends_ratio'] = users_df['followers_count'] / users_df['friends_count']
users_df['statuses_favourites_ratio'] = users_df['statuses_count'] / users_df['favourites_count']
users_df['statuses_per_month'] = users_df['statuses_count'] / users_df['months']
users_df['favourites_per_month'] = users_df['favourites_count'] / users_df['months']


