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

seed = 42
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

# The outliers won't enable any real insight to be taken from the distribution of users, so let's take a look at a clean version of the data.

thresh = 0.95
outliers_mask = (users_df.friends_count < users_df.friends_count.quantile(thresh)) &\
                (users_df.followers_count < users_df.followers_count.quantile(thresh))

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

users_df[outliers_mask].friends_count.plot(kind='box', ax=axes[0], title='Friends Count Boxplot')
users_df[outliers_mask].followers_count.plot(kind='box', ax=axes[1], title='Followers Count Bocplot');
# -

users_df.loc[outliers_mask, ['friends_count', 'followers_count']].describe()

# Since the two distributions are heavily skewed, we shall user the median as a representative of the average number of friends and followers, and it is observed after the data is cleaned that the median friends and followers count is 400 and 489, while the median count before cleaning was 430 and 658.

print('Number of outliers:', users_df.shape[0] - users_df[outliers_mask].shape[0])
print('Outliers friends and followers summary:')
users_df.loc[~outliers_mask, ['friends_count', 'followers_count']].describe()

# The condition that I chose for the setting the outliers mask depended that both friends and followers counts weren't outliers, therefore the minimum followers count in the outliers is less than the maximum followers count in the clean version of the data.

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

users_df[~outliers_mask].friends_count.plot(kind='hist', ax=axes[0], bins=50, title='Friends Count Boxplot')
users_df[~outliers_mask].followers_count.plot(kind='hist', ax=axes[1], bins=50, title='Followers Count Boxplot');
# -

# new addition
sns.histplot(data=users_df.query('friends_count > 0'), x='friends_count', log_scale=True);

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

# There is some sort of linear relationship going on below 25000 followers. I guess that these are users that get followers by following other users in return of these users following them back. That's why these have an apparent linear relationship between friends and followers count. 
#
# On the other hand we can see the other distribution of users who don't show a linear relationship between these two features, and these may be the users who have amassed a following based on their activity, and not relying on quid pro quo agreement with other users to follow them and expect a follow back.

# ## Clustering users based on friends and followers count

# +
# Kmeans clustering
from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k)

y_pred = kmeans.fit_predict(users_df.loc[outliers_mask, ['friends_count', 'followers_count']])

# +
color_mapping = {cluster: color for cluster, color in zip(set(y_pred), 'bgrcmy')}
plt.figure(figsize=(12, 8))

for cluster, color in color_mapping.items():
    plt.scatter(x=users_df[outliers_mask][y_pred == cluster]['friends_count'],
                y=users_df[outliers_mask][y_pred == cluster]['followers_count'],
                c=color,
                alpha=0.5,
                label=cluster);

plt.legend();
# -

users_df[outliers_mask][y_pred == 0].plot(kind='scatter',
                             x='friends_count',
                             y='followers_count',
                             figsize=(10, 6),
                             alpha=0.5);


# In the first cluster we can visualize a linear relationship between friends and followers count in users with less than 10000 followers, where that represents a cutoff and the relationship then ceases to exist in further clusters.

users_df[outliers_mask][y_pred != 0].plot(kind='scatter',
                             x='friends_count',
                             y='followers_count',
                             figsize=(10, 6),
                             c=y_pred[y_pred != 0],
                             cmap='viridis');

# Passing on to the rest of the clusters we can see that what distinguishes each cluster from another is that width of the friends count distribution gets narrower and narrower, with less and and less outliers, indicating that the majority of users who have a large following base didn't get there by the quid pro quo way that some of the users in cluster 0 did.

# +
k = 2
kmeans = KMeans(n_clusters=k)

y_pred_2 = kmeans.fit_predict(users_df[outliers_mask][y_pred == 0][['friends_count', 'followers_count']])

for label, color in zip(set(y_pred_2), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred == 0][y_pred_2 == label]['friends_count'],
                y=users_df[outliers_mask][y_pred == 0][y_pred_2 == label]['followers_count'],
                c=color,
                alpha=0.5,
                label=label);

plt.legend();    

# +
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.2, min_samples=5)

dbscan.fit(users_df[outliers_mask][y_pred == 0][['friends_count', 'followers_count']])

for label, color in zip(set(dbscan.labels_), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred == 0][dbscan.labels_ == label]['friends_count'],
                y=users_df[outliers_mask][y_pred == 0][dbscan.labels_ == label]['followers_count'],
                c=color,
                alpha=0.5,
                label=label);

plt.legend();    

# +
from sklearn.mixture import GaussianMixture

n = 3
gm = GaussianMixture(n_components=n, n_init=10, covariance_type='tied')
gm.fit(users_df[outliers_mask][y_pred == 0][['friends_count', 'followers_count']])

y_pred_3 = gm.predict(users_df[outliers_mask][y_pred == 0][['friends_count', 'followers_count']])

for label, color in zip(set(y_pred_3), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred == 0][y_pred_3 == label]['friends_count'],
                y=users_df[outliers_mask][y_pred == 0][y_pred_3 == label]['followers_count'],
                c=color,
                alpha=0.5,
                label=label);

plt.legend();    

# +
n = 5
gm = GaussianMixture(n_components=n, n_init=10, covariance_type='tied')
gm.fit(users_df[outliers_mask][['friends_count', 'followers_count']])

y_pred_4 = gm.predict(users_df[outliers_mask][['friends_count', 'followers_count']])

plt.figure(figsize=(10, 6))

for label, color in zip(set(y_pred_4), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred_4 == label]['friends_count'],
                y=users_df[outliers_mask][y_pred_4 == label]['followers_count'],
                c=color,
                alpha=0.5,
                label=label);

plt.legend();
# -

# I think that gaussian mixture model nailed the clustering better than kmeans.

users_df[outliers_mask][y_pred_4 == 0].plot(kind='scatter',
                             x='friends_count',
                             y='followers_count',
                             figsize=(10, 6),
                             alpha=0.1);

users_df[outliers_mask][y_pred_4 != 0].plot(kind='scatter',
                             x='friends_count',
                             y='followers_count',
                             figsize=(10, 6),
                             alpha=0.5);

# I still believe that this cluster can be further split into two cluster, and I could do that manually to further look into users who I think used the follow and follow back scheme of gathering followers.

cluster_0 = users_df[outliers_mask][y_pred_4 == 0]
thresh = 0.8
cluster_0_mask = (cluster_0['friends_count'] / (cluster_0['followers_count'] + 0.0001)) > thresh
users_df[outliers_mask][y_pred_4 == 0][cluster_0_mask].plot(kind='scatter',
                             x='friends_count',
                             y='followers_count',
                             figsize=(10, 6),
                             alpha=0.2);

# Since the threshold is a hyperparamter in this expirement of analyzing users who might only have followers because they followed them in the first place, I'll analyze the whole cluster with no threshold, and with two thresholds to have a broader look into the data.

# ## Analyze time present for cluster 0

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


# -

def split_cluster_0(thresh):
    cluster_0_mask = (cluster_0['friends_count'] / (cluster_0['followers_count'] + 0.0001)) > thresh
    return cluster_0[cluster_0_mask], cluster_0[~cluster_0_mask]


cluster_0 = users_df[outliers_mask][y_pred_4 == 0]
cluster_0_1, _ = split_cluster_0(0.4)
cluster_0_2, _ = split_cluster_0(0.8)

# +
# cluster_0.plot(kind='scatter',
#              x='friends_count',
#              y='followers_count',
#              figsize=(10, 6),
#              alpha=0.1,
#              s=cluster_0.months);

plt.figure(figsize=(10, 6))

plt.scatter(x=cluster_0['friends_count'],
            y=cluster_0['followers_count'],
            c=cluster_0['months'],
            alpha=0.5);
#             cmap='viridis_r');
plt.colorbar();
# -

# The months since account creation helped in further illuminating the differences between users in this cluster, and it shows the two clusters withing cluster 0, where some of the users have large number of followers and friends, and have not joined twitter since too long.
#
# I guess that this could further be illuminated if we added statuses count into the plot.

# +
plt.figure(figsize=(10, 6))

plt.scatter(x=cluster_0['friends_count'],
            y=cluster_0['followers_count'],
            c=cluster_0['months'],
            s=cluster_0['statuses_count']/1000,
            alpha=0.5)
#             cmap='viridis_r');
plt.colorbar();
# -

# This addition just made it really interesting, as it could be easily seen that the users that have low friends count and high followers count have high statuses count and have been on twitter for longer than the other category of the users, which have gained followers by follow back schemes and haven't been on twitter for so long.

# +
plt.figure(figsize=(10, 6))

plt.scatter(x=cluster_0['friends_count'],
            y=cluster_0['followers_count'],
            c=cluster_0['months'],
            s=cluster_0['favourites_count']/1000,
            alpha=0.5)
#             cmap='viridis_r');
plt.colorbar();
# -

# From that perspective we can also see that the two categories of users differ in their favorties count. Users that have been long on twitter and have high followers and low friends count tend to have lower favourties count than users who haven't been on twitter for long and used follo back schemes to gain followers.
#
#
# Now it would be interesting to test clustering users based on all of the features that we have visualized so far.

# +
k = 5
kmeans = KMeans(n_clusters=k)
cols = ['friends_count', 'followers_count', 'statuses_count', 'favourites_count', 'months']

y_pred = kmeans.fit_predict(users_df.loc[outliers_mask, cols])

color_mapping = {cluster: color for cluster, color in zip(set(y_pred), 'bgrcmy')}
plt.figure(figsize=(10, 6))

for cluster, color in color_mapping.items():
    plt.scatter(x=users_df[outliers_mask][y_pred == cluster]['friends_count'],
                y=users_df[outliers_mask][y_pred == cluster]['followers_count'],
                c=color,
                alpha=0.5,
                label=cluster);

plt.legend();

# +
for label, color in zip(set(y_pred), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred == label]['friends_count'],
                y=users_df[outliers_mask][y_pred == label]['followers_count'],
                c=color,
                s=users_df[outliers_mask][y_pred == label]['favourites_count']/1000,
                alpha=0.5,
                label=label);

plt.legend();
# -

# The results don't look good at all, since the algorithm isn't able to cluster user groups on any dimension.

# +
n = 10
gm = GaussianMixture(n_components=n, n_init=10, random_state=44)
gm.fit(users_df[outliers_mask][cols])

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
plt.figure(figsize=(8, 6))
for label, color in zip(set(y_pred_4), 'bgrcmy'):
    plt.scatter(x=users_df[outliers_mask][y_pred_4 == label]['friends_count'],
                y=users_df[outliers_mask][y_pred_4 == label]['followers_count'],
                c=color,
                s=users_df[outliers_mask][y_pred_4 == label]['favourites_count']/100,
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


def scatter_cluster(tmp, scale=1):
    plt.figure(figsize=(10, 6))
    plt.scatter(x=tmp['friends_count'],
                y=tmp['followers_count'],
                s=tmp['statuses_count']/scale,
                c=tmp['months'])
    plt.colorbar()


scatter_cluster(gm_0, 100)
gm_0[cols].describe()

sns.pairplot(gm_0[cols], height=1.5);

scatter_cluster(gm_1, 100)
gm_1[cols].describe()

sns.pairplot(gm_1[cols], height=1.5);

# The first two cluster represent two types of inactive users, and those are users who have used twitter for a while judging by their activity, and users who didn't use twitter as much. These two cluster could be merged into a single cluster for the inactive users.

inactive_users = pd.concat([gm_0, gm_1])
scatter_cluster(inactive_users, 100)
inactive_users[cols].describe()

# Looking into the single variable distributions of this group be can see.

# +
import seaborn as sns

sns.pairplot(inactive_users[cols], height=1.5);
# -

# The distribution of months present for inactive users in right skewed, indicating that the majority of them have recently joined twitter.

scatter_cluster(gm_2, 100)
gm_2[cols].describe()

# This cluster represents influencer users that have been on twitter for an average of 9.5 years.

sns.pairplot(gm_2[cols], height=1.5);

# The distribution of months present appears to be normal, and centered around 116 months. The minimum months present in this cluster is 75, which indicates that this clusters contains only veteran users.
#
# TODO: look into followers distribution more closely.
#
# There looks to be an obvious positive correlation between months and statuses count, and a negative correlation could be present between statuses and favourites count. There is also positive correlation between statuses and followers count, indicating that users with more statuses tend to have more followers in this cluster.

scatter_cluster(gm_3, 100)
gm_3[cols].describe()

# This cluster also represents influencer users that have been on twitter for a period less than the other group, with an average of 7 years. They also have significantly less number of average statuses.
#
# The fading effect observed with increasing of point size indicates that spme of the users on the left might not be personal accounts, but rather official accounts.

sns.pairplot(gm_3[cols], height=1.5);

# We can immediately notice in this plot that the distribution of statuses count isn't normal, and this cluster's users are generally less active than the previous cluster, or they most probably they just have joined later than the previous cluster's users. They don't exhibit the relationship between statuses count and months.
#
# This clusters' users tend to have significantly less favourite counts than the other group.
#
# The distribution of months present is normal with a left skew, and the minimum is 1 month, which indicates that this cluster includes users whose accounts were maybe deleted so they made new accounts, offical accounts representing entities, or anomalous users who may have gained a huge following in a short amount of time for some reason.

scatter_cluster(gm_4, 100)
gm_4[cols].describe()

# This is the cluster that I've been looking for, and it contains the users that have low friend to follower ratio. These users haven't been on twitter for so long, and have managed to get high follower count by using follow back schemes with other users. They tend to have less statuses than favorites count, and that maybe be because they are more observers than influencers on twitter. I would call this cluster the average joes of twitter.

sns.pairplot(gm_4[cols], height=1.5);

# We can see that obvious linear relationship between the followers and friends count in this cluster, which wasn't present in any other cluster. We can also see the distribution of months present isn't normally distirbuted and rather right skewed, indicating that these users tend to be users who joined in the past 5 years or so.

scatter_cluster(gm_5, 100)
gm_5[cols].describe()

# I think that this cluster represents another version of the average joes, and these are the less active average joes of twitter. A characterstic of this group is that they have less followers than friends on average.

sns.pairplot(gm_5[cols], height=1.5);

# +
# Now it's time to save these results to enable further analysis based on them
import os
import sys
import pickle as pk

from datetime import datetime


def save_model(model, filename, save_dir='models'):
    filename = '{}_{}.pkl'.format(filename, datetime.now())
    filepath = os.path.join(save_dir, filename)
    file = open(filepath+'.pkl', 'wb')
    
    pk.dump(model, file)
    print('Model saved as {} at {}/'.format(filename, save_dir) , file=sys.stderr)
    
def load_model(filename, save_dir='models'):
    filepath = os.path.join(save_dir, filename)
    file = open(filepath+'.pkl', 'rb')
    
    model = pk.load(file)
    print('{} loaded from {}'.format(filename, save_dir), file=sys.stderr)


# -

save_model(gm, 'gm')

gm__2 = load_model('gm_2021-04-07 12:55:26.141024.pkl')

# +
# new addition

# months distribution
sns.histplot(data=gm_2.query('months > 0'), bins=10, x='months');

# +
# new addition

# statuses count distribution
sns.histplot(data=users_df.query('statuses_count > 0'), x='statuses_count', log_scale=True);

# statuses count distribution is log normal

# +
# new addition

# favourite count distribution
sns.histplot(data=users_df.query('favourites_count > 0'), x='favourites_count', log_scale=True);

# the distribution of favourties is also log normal

# +
# new addition

# statuses and favourites
splot = sns.regplot(data=users_df.query('favourites_count > 0'),
                    x='statuses_count',
                    y='favourites_count',
                    fit_reg=False)
splot.set(xscale='log', yscale='log');

# +
# new addition

# followers distribution
sns.histplot(data=gm_2.query('followers_count > 0'), x='followers_count');

# the distribution is log normal

# +
# new addition

# friends distribution
sns.histplot(data=gm_1.query('friends_count > 0'), x='friends_count', log_scale=True);

# the distribution is log normal with a left skew

# +
# new addition

splot = sns.scatterplot(data=users_df[outliers_mask],
             x='friends_count',
             y='followers_count',
             alpha=0.2)

# -


